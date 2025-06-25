import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import hashlib
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FigmaNodeSimilarityDetector:
    """
    A flexible similarity detection system for Figma nodes using bottom-up approach.
    Feature vectors are extracted directly from leaf nodes and propagated upward.
    """
    
    def __init__(self, similarity_threshold: float = 0.8, use_semantic_embeddings: bool = True):
        """
        Initialize the similarity detector
        
        Args:
            similarity_threshold: Threshold for considering nodes similar (0-1)
            use_semantic_embeddings: Whether to use semantic embeddings for text content
        """
        self.similarity_threshold = similarity_threshold
        self.use_semantic_embeddings = use_semantic_embeddings
        self.node_feature_vectors = {}  # Store feature vectors directly
        self.node_metadata = {}  # Store metadata (is_leaf, etc.)
        self.similarity_matrix = None
        self.clusters = None
        self.node_tree = {}  # Store hierarchical structure
        
        # Initialize semantic model if needed
        if use_semantic_embeddings:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded semantic embedding model")
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}")
                self.use_semantic_embeddings = False
    
    def extract_leaf_node_feature_vector(self, node_data: Dict[str, Any], node_path: str = "") -> np.ndarray:
        """
        Extract feature vector directly from a leaf Figma node

        Args:
            node_data: The node data dictionary
            node_path: Path to the node in the tree

        Returns:
            Numpy array containing the feature vector
        """
        node_info = node_data.get('node', {})
        vector = []

        # 1. Structural Features (normalized)
        node_type = node_info.get('type', '')
        tag = node_data.get('tag', '')
        has_children = len(node_data.get('children', [])) > 0
        num_children = len(node_data.get('children', []))
        node_layout = node_info.get('layout', '')
        vector.extend([
            hash(node_type) % 1000 / 1000.0,  # Normalize hash
            # hash(tag) % 1000 / 1000.0,
            float(has_children),
            min(num_children / 10000.0, 1.0),  # Normalize to 0-1
            hash(node_layout) % 1000 / 1000.0,  # Normalize hash
            (node_info.get('width', 0) * node_info.get('width', 0)) % 100000 / 100000.0 * 10
        ])

        # 2. Style Features
        # Extract font information from textStyle
        text_style = node_info.get('textStyle', {})
        font_family = text_style.get('fontFamily', '')
        font_size = text_style.get('fontSize', 0)
        font_style = text_style.get('fontStyle', '')
        font_weight = text_style.get('fontWeight', 0)

        # Extract fill colors
        fills = node_info.get('fills', [])
        fill_r = fill_g = fill_b = fill_a = 0
        if fills:
            primary_fill = fills[0]
            color = primary_fill.get('color', {})
            fill_r = color.get('r', 0)
            fill_g = color.get('g', 0)
            fill_b = color.get('b', 0)
            fill_a = color.get('a', 1)

        # Extract stroke colors
        strokes = node_info.get('strokes', [])
        stroke_r = stroke_g = stroke_b = stroke_a = 0
        if strokes:
            primary_stroke = strokes[0]
            stroke_color = primary_stroke.get('color', {})
            stroke_r = stroke_color.get('r', 0)
            stroke_g = stroke_color.get('g', 0)
            stroke_b = stroke_color.get('b', 0)
            stroke_a = stroke_color.get('a', 1)

        vector.extend([
            hash(font_family) % 1000 / 1000.0 if font_family else 0,
            min(font_size / 10000.0, 1.0) if font_size else 0,  # Normalize font size to 0-1 (assuming max ~100px)
            hash(font_style) % 1000 / 1000.0 if font_style else 0,
            min(font_weight / 10000.0, 1.0) if font_weight else 0,  # Normalize font weight (max ~900-1000)
            fill_r, fill_g, fill_b, fill_a,
            stroke_r, stroke_g, stroke_b, stroke_a,
        ])

        # 3. Content Features
        text_content = node_info.get('characters', '')
        name = node_data.get('name', '')
        has_text = bool(text_content)

        vector.extend([
            float(has_text),
        ])

        # 4. Semantic Features (using embeddings) - commented out for now
        # if self.use_semantic_embeddings:
        #     text_for_embedding = f"{name} {node_type}"
        #     if text_for_embedding.strip():
        #         try:
        #             semantic_features = self.semantic_model.encode([text_for_embedding])[0].tolist()
        #             vector.extend(semantic_features)
        #         except:
        #             vector.extend([0.0] * 384)  # Default embedding size
        #     else:
        #         vector.extend([0.0] * 384)
        # else:
        #     vector.extend([0.0] * 384)

        # 5. Descendant Count Feature (for leaf nodes, this is 0)
        total_descendants = 0  # Leaf nodes have no descendants
        vector.extend([
            total_descendants
        ])

        return np.array(vector)


    def _calculate_area_weights(self, parent_data: Dict[str, Any], children_data: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate area-based weights for children nodes relative to parent.

        Args:
            parent_data: Parent node data
            children_data: List of children node data

        Returns:
            List of normalized weights (sum to 1.0) or None if calculation fails
        """
        if not children_data:
            return None

        try:
            # Extract parent dimensions
            parent_node = parent_data.get('node', {})
            parent_width = parent_node.get('width', 0)
            parent_height = parent_node.get('height', 0)
            parent_area = parent_width * parent_height

            if parent_area <= 0:
                logger.warning("Parent area is zero or negative, using equal weights")
                return [1.0 / len(children_data)] * len(children_data)

            # Calculate children areas
            children_areas = []
            for child_data in children_data:
                child_node = child_data.get('node', {})
                child_width = child_node.get('width', 0)
                child_height = child_node.get('height', 0)
                child_area = child_width * child_height
                children_areas.append(child_area)

            # Calculate weights as ratio of child area to parent area
            weights = [child_area / parent_area for child_area in children_areas]

            # Normalize weights to sum to 1.0
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
            else:
                # If all areas are zero, use equal weights
                normalized_weights = [1.0 / len(children_data)] * len(children_data)

            logger.debug(f"Calculated area weights: {normalized_weights}")
            return normalized_weights

        except Exception as e:
            logger.warning(f"Error calculating area weights: {e}")
            return None
        
    def create_parent_feature_vector(self, children_vectors: List[np.ndarray], node_data: Dict[str, Any], children_data: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Create parent node feature vector by computing weighted average of children feature vectors
        based on the area ratio of each child to the parent node.

        Args:
            children_vectors: List of children feature vectors
            node_data: Parent node data
            children_data: List of children node data (needed for area calculations)

        Returns:
            Feature vector for parent node
        """
        if not children_vectors:
            # If no children vectors, create a default vector
            default_size = 19  # structural(4) + style(12) + content(1) + descendant(1)
            return np.zeros(default_size)

        # Calculate areas and weights
        weights = self._calculate_area_weights(node_data, node_data.get('children', {}))
        
        if weights is None or len(weights) != len(children_vectors):
            # Fallback to simple average if area calculation fails
            logger.warning("Area calculation failed, falling back to simple average")
            parent_vector = np.mean(children_vectors, axis=0)
        else:
            # Calculate weighted average
            weighted_vectors = []
            for i, (vector, weight) in enumerate(zip(children_vectors, weights)):
                weighted_vectors.append(vector * weight)

            parent_vector = np.sum(weighted_vectors, axis=0)
            logger.debug(f"Applied area weights: {weights}")

        # Override some structural features specific to the parent
        node_info = node_data.get('node', {})
        node_type = node_info.get('type', '')
        tag = node_data.get('tag', '')
        num_children = len(children_vectors)
        node_layout = node_info.get('layout', '')

        # Update structural features (first 4 elements)
        parent_vector[0] = hash(node_type) % 1000 / 1000.0
        parent_vector[1] = 1.0  # Parent always has children
        parent_vector[2] = min(num_children / 10000.0, 1.0)
        parent_vector[3] = hash(node_layout) % 1000 / 1000.0
        parent_vector[4] = (node_info.get('width', 0) * node_info.get('width', 0)) % 100000000 / 100000000.0 * 10

        # Calculate total descendants
        total_descendants = num_children
        for child_vector in children_vectors:
            # The descendant count is the last feature in the vector
            child_descendants = int(child_vector[-1] * 10000)  # Denormalize
            total_descendants += child_descendants

        # Update the descendant count feature (last element)
        parent_vector[-1] = min(total_descendants / 10000.0, 1.0)

        return parent_vector


    
    def build_similarity_matrix(self, nodes_data: Dict[str, Any]) -> np.ndarray:
        """
        Build similarity matrix using bottom-up approach.
        Feature vectors are extracted from leaves and propagated upward.
        
        Args:
            nodes_data: Dictionary containing all node data
            
        Returns:
            Similarity matrix as numpy array
        """
        # Step 1: Build the tree structure and identify leaf nodes
        self.node_tree = self._build_node_tree(nodes_data)
        
        # Step 2: Extract feature vectors bottom-up
        self.node_feature_vectors = {}
        self.node_metadata = {}
        self._extract_feature_vectors_bottom_up(nodes_data)
        
        # Step 3: Filter out leaf nodes for similarity calculation (only compare non-leaf nodes)
        non_leaf_paths = [path for path, metadata in self.node_metadata.items() if not metadata['is_leaf']]
        self.node_paths = non_leaf_paths
        
        if not self.node_paths:
            logger.warning("No non-leaf nodes found for similarity comparison")
            return np.array([])
        
        # Step 4: Get feature vectors for non-leaf nodes only
        feature_vectors = []
        for node_path in self.node_paths:
            vector = self.node_feature_vectors[node_path]
            feature_vectors.append(vector)
        
        feature_vectors = np.array(feature_vectors)
        
        # Step 5: Calculate similarity matrix using cosine similarity
        self.similarity_matrix = cosine_similarity(feature_vectors)
        
        return self.similarity_matrix
    
    def _build_node_tree(self, data: Dict[str, Any], path: str = "", parent_path: str = None) -> Dict[str, Dict]:
        """Build a tree structure mapping node paths to their metadata"""
        tree = {}
        
        if isinstance(data, dict):
            if 'node' in data:  # This is a node
                current_path = f"{path}/{data.get('name', 'unnamed')}" if path else data.get('name', 'root')
                
                tree[current_path] = {
                    'data': data,
                    'parent': parent_path,
                    'children': [],
                    'is_leaf': len(data.get('children', [])) == 0
                }
                
                if 'children' in data:
                    for child in data['children']:
                        child_tree = self._build_node_tree(child, current_path, current_path)
                        tree.update(child_tree)
                        # Add child paths to current node
                        for child_path in child_tree.keys():
                            if child_tree[child_path]['parent'] == current_path:
                                tree[current_path]['children'].append(child_path)
        
        return tree
    
    def _extract_feature_vectors_bottom_up(self, data: Dict[str, Any], path: str = ""):
        """Extract feature vectors using bottom-up approach"""
        if isinstance(data, dict):
            if 'node' in data:  # This is a node
                current_path = f"{path}/{data.get('name', 'unnamed')}" if path else data.get('name', 'root')
                
                # First, process all children
                children_vectors = []
                if 'children' in data:
                    for child in data['children']:
                        child_path = f"{current_path}"
                        self._extract_feature_vectors_bottom_up(child, child_path)
                        
                        # Get child path and vector
                        child_node_path = f"{current_path}/{child.get('name', 'unnamed')}"
                        if child_node_path in self.node_feature_vectors:
                            children_vectors.append(self.node_feature_vectors[child_node_path])
                
                # Extract feature vector for current node
                is_leaf = len(data.get('children', [])) == 0
                
                if is_leaf:  # Leaf node
                    self.node_feature_vectors[current_path] = self.extract_leaf_node_feature_vector(data, current_path)
                else:  # Parent node - aggregate children vectors
                    self.node_feature_vectors[current_path] = self.create_parent_feature_vector(children_vectors, data)
                
                # Store metadata
                self.node_metadata[current_path] = {
                    'is_leaf': is_leaf,
                    'num_children': len(children_vectors),
                    'node_type': data.get('node', {}).get('type', ''),
                    'name': data.get('name', '')
                }
    
    def calculate_group_average_vectors(self, similarity_groups: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Calculate average feature vector for each similarity group
        
        Args:
            similarity_groups: Dictionary mapping group_id to list of node paths
            
        Returns:
            Dictionary mapping group_id to average feature vector
        """
        group_vectors = {}
        
        for group_id, node_paths in similarity_groups.items():
            group_feature_vectors = []
            for node_path in node_paths:
                if node_path in self.node_feature_vectors:
                    group_feature_vectors.append(self.node_feature_vectors[node_path])
            
            if group_feature_vectors:
                # Calculate average vector for the group
                group_vectors[group_id] = np.mean(group_feature_vectors, axis=0)
            else:
                logger.warning(f"No feature vectors found for group {group_id}")
        
        return group_vectors

    def merge_similar_groups(self, similarity_groups: Dict[str, List[str]], 
                            merge_threshold: float = 0.9) -> Dict[str, List[str]]:
        """
        Merge groups that have similar average feature vectors

        Args:
            similarity_groups: Original similarity groups
            merge_threshold: Threshold for merging groups (default 0.9)

        Returns:
            Merged similarity groups
        """
        if not similarity_groups:
            return similarity_groups

        # Calculate average vectors for each group
        group_vectors = self.calculate_group_average_vectors(similarity_groups)

        if len(group_vectors) <= 1:
            return similarity_groups

        # Create similarity matrix between group average vectors
        group_ids = list(group_vectors.keys())
        group_vector_matrix = np.array([group_vectors[group_id] for group_id in group_ids])
        group_similarity_matrix = cosine_similarity(group_vector_matrix)

        logger.warning(f"group_vector_matrix: {group_vector_matrix}")


        # Find groups to merge based on similarity threshold
        merged_groups = {}
        processed_groups = set()
        merge_counter = 0

        for i, group_id_i in enumerate(group_ids):
            if group_id_i in processed_groups:
                continue

            # Find all groups similar to current group
            similar_group_indices = np.where(group_similarity_matrix[i] >= merge_threshold)[0]
            groups_to_merge = [group_ids[j] for j in similar_group_indices]

            if len(groups_to_merge) > 1:  # If we found groups to merge
                # Create new merged group
                merged_group_id = f"merged_group_{merge_counter}"
                merged_nodes = []

                for group_to_merge in groups_to_merge:
                    merged_nodes.extend(similarity_groups[group_to_merge])
                    processed_groups.add(group_to_merge)

                merged_groups[merged_group_id] = merged_nodes
                merge_counter += 1

                logger.info(f"Merged groups {groups_to_merge} into {merged_group_id}")
            else:
                # Keep original group if no merging needed
                merged_groups[group_id_i] = similarity_groups[group_id_i]
                processed_groups.add(group_id_i)

        return merged_groups
    
    # Modified check_similarity function
    def check_similarity(self, threshold: float = None, merge_threshold: float = 0.9, 
                        enable_group_merging: bool = False) -> Dict[str, List[str]]:
        """
        Check similarity between non-leaf nodes and return groups of similar nodes.
        Optionally merge similar groups based on their average feature vectors.

        Args:
            threshold: Similarity threshold for initial grouping (uses instance threshold if None)
            merge_threshold: Threshold for merging groups (default 0.9)
            enable_group_merging: Whether to enable group merging stage

        Returns:
            Dictionary mapping group_id to list of similar node paths (non-leaf nodes only)
        """
        if self.similarity_matrix is None:
            raise ValueError("Must build similarity matrix first")

        if len(self.similarity_matrix) == 0:
            logger.warning("No non-leaf nodes available for similarity comparison")
            return {}

        threshold = threshold or self.similarity_threshold

        # Step 1: Initial grouping based on node similarity
        similarity_groups = self._threshold_based_grouping(threshold)
        logger.info(f"Initial grouping created {len(similarity_groups)} groups")

        # Step 2: Merge similar groups if enabled
        if enable_group_merging and len(similarity_groups) > 1:
            merged_groups = self.merge_similar_groups(similarity_groups, merge_threshold)
            logger.info(f"After merging: {len(merged_groups)} groups remain")
            return merged_groups

        return similarity_groups
    
    def _threshold_based_grouping(self, threshold: float) -> Dict[str, List[str]]:
        """Threshold-based similarity grouping for non-leaf nodes only"""
        groups = {}
        assigned = set()
        group_counter = 0
        
        for i in range(len(self.node_paths)):
            if self.node_paths[i] in assigned:
                continue
                
            # Find all nodes similar to current node
            similar_indices = np.where(self.similarity_matrix[i] >= threshold)[0]
            similar_nodes = [self.node_paths[j] for j in similar_indices if j != i]
            
            if similar_nodes:  # If we found similar nodes
                group_id = f"group_{group_counter}"
                groups[group_id] = [self.node_paths[i]] + similar_nodes
                assigned.update(groups[group_id])
                group_counter += 1
        
        return groups
    
    def add_node_ids_to_json(self, original_data: Dict[str, Any], similarity_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Add node_id to each node in the original JSON based on similarity groups.
        Only non-leaf nodes get group IDs, leaf nodes get unique IDs.
        
        Args:
            original_data: Original JSON data
            similarity_groups: Groups of similar nodes (non-leaf only)
            
        Returns:
            Modified JSON with node_id added to each node
        """
        # Create mapping from node path to group id
        path_to_group = {}
        for group_id, node_paths in similarity_groups.items():
            for node_path in node_paths:
                path_to_group[node_path] = group_id
        
        # Add node_ids recursively
        modified_data = self._add_node_ids_recursive(original_data, path_to_group)
        
        return modified_data
    
    def _add_node_ids_recursive(self, data: Dict[str, Any], path_to_group: Dict[str, str], path: str = "") -> Dict[str, Any]:
        """Recursively add node_ids to the JSON structure"""
        if isinstance(data, dict):
            result = data.copy()
            
            if 'node' in data:  # This is a node
                current_path = f"{path}/{data.get('name', 'unnamed')}" if path else data.get('name', 'root')
                
                # Check if this is a leaf node
                is_leaf = len(data.get('children', [])) == 0
                
                if is_leaf:
                    # Leaf nodes get unique IDs
                    result['node_id'] = f"leaf_{hashlib.md5(current_path.encode()).hexdigest()[:8]}"
                else:
                    # Non-leaf nodes get group IDs if they're in a similarity group
                    if current_path in path_to_group:
                        result['node_id'] = path_to_group[current_path]
                    else:
                        # Generate unique ID for non-grouped non-leaf nodes
                        result['node_id'] = f"unique_{hashlib.md5(current_path.encode()).hexdigest()[:8]}"
            
            if 'children' in data:
                child_path = f"{path}/{data.get('name', 'unnamed')}" if path else data.get('name', 'root')
                result['children'] = [
                    self._add_node_ids_recursive(child, path_to_group, child_path)
                    for child in data['children']
                ]
            
            return result
        
        return data

    def get_feature_vector_info(self) -> Dict[str, Any]:
        """
        Get information about the feature vectors for debugging/analysis
        
        Returns:
            Dictionary containing feature vector statistics
        """
        if not self.node_feature_vectors:
            return {"error": "No feature vectors extracted yet"}
        
        vector_lengths = [len(v) for v in self.node_feature_vectors.values()]
        leaf_count = sum(1 for metadata in self.node_metadata.values() if metadata['is_leaf'])
        non_leaf_count = len(self.node_metadata) - leaf_count
        
        return {
            "total_nodes": len(self.node_feature_vectors),
            "leaf_nodes": leaf_count,
            "non_leaf_nodes": non_leaf_count,
            "feature_vector_length": vector_lengths[0] if vector_lengths else 0,
            "feature_breakdown": {
                "structural_features": 3,
                "style_features": 12,  # Updated: font_family, font_size, font_style, font_weight + colors
                "content_features": 1,
                "descendant_count": 1,
                "semantic_features": 384 if self.use_semantic_embeddings else 0
            }
        }

    def print_figma_tree_with_vectors(self, node, depth=0, path=""):
        """
        Print the Figma node tree with feature vectors
        
        Args:
            node: The current node to print
            depth: Current depth in the tree
            path: Current path to the node
        """
        indent = "  " * depth  # 2 spaces per level

        # Extract info
        name = node.get("name", "[no name]")
        tag = node.get("tag", "[no tag]")
        node_id = node.get("node_id", "")
        
        # Handle TEXT nodes with characters
        node_data = node.get("node", {})
        characters = node_data.get("characters", "")
        is_text = tag == "TEXT"
        display_name = characters[:10] + "..." if is_text and characters else name

        # Layout info (if present)
        layout = node_data.get("layoutMode", "NONE")
        layout_str = "ROWS" if layout == "HORIZONTAL" else "COLS" if layout == "VERTICAL" else layout

        # Get the current node path
        current_path = f"{path}/{name}" if path else name
        
        # Get feature vector if available
        vector_str = ""
        if current_path in self.node_feature_vectors:
            vector = self.node_feature_vectors[current_path]
            # Format vector to show first few values and some key features
        if len(vector) > 0:
            vector_str = f" | Vector: [{', '.join(f'{v:.7f}' for v in vector)}] (len={len(vector)})"
        else:
            vector_str = " | Vector: Not found"

        # Print current node info with feature vector
        print(f"{indent}- {display_name} [{tag}] -> {name} {layout_str} ({node_id}){vector_str}")

        # Recursively print children
        for child in node.get("children", []):
            self.print_figma_tree_with_vectors(child, depth + 1, current_path)






app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5123"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class JSONData(BaseModel):
    data: Dict[str, Any]

class SimilarityResponse(BaseModel):
    processed_data: Dict[str, Any]
    similarity_groups: Dict[str, Any]  # Replace with your actual similarity groups type

def rename_names_and_store_mapping(node, counter, mapping):
    """Recursively rename 'name' fields and store mapping"""
    if isinstance(node, dict):
        if "name" in node:
            old_name = node["name"]
            new_name = f"name{counter[0]}"
            mapping[new_name] = old_name
            node["name"] = new_name
            counter[0] += 1
        for key in node:
            rename_names_and_store_mapping(node[key], counter, mapping)
    elif isinstance(node, list):
        for item in node:
            rename_names_and_store_mapping(item, counter, mapping)

def restore_names_from_mapping(node, mapping):
    """Recursively restore original names from mapping"""
    if isinstance(node, dict):
        if "name" in node and node["name"] in mapping:
            node["name"] = mapping[node["name"]]
        for key in node:
            restore_names_from_mapping(node[key], mapping)
    elif isinstance(node, list):
        for item in node:
            restore_names_from_mapping(item, mapping)


        

with open("PAGE_109.json", "r") as f:
    data = json.load(f)

# Rename names and create mapping
name_mapping = {}
counter = [0]
rename_names_and_store_mapping(data, counter, name_mapping)

# Process with similarity detector
detector = FigmaNodeSimilarityDetector(
    similarity_threshold=0.99999999999,
    use_semantic_embeddings=True
)

similarity_matrix = detector.build_similarity_matrix(data)
similarity_groups = detector.check_similarity(
    threshold=None,  # Use default threshold
    merge_threshold=0.9999,  # Threshold for merging groups
    enable_group_merging=True  # Enable the merging stage
)
result_json = detector.add_node_ids_to_json(data, similarity_groups)

detector.print_figma_tree_with_vectors(result_json)

# Restore original names
restore_names_from_mapping(result_json, name_mapping)


with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False)


