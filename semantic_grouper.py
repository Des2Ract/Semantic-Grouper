import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import hashlib
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import re
import copy

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import json
import random
from typing import Dict, List, Any

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
        tag = node_data.get('tag', '')
        
        # Check if tag is ICON or SVG - if so, return default vector of ones
        if tag in ["ICON", "SVG"]:
            # Create default vector with ones
            default_size = 17  # structural(3) + style(12) + content(1) + descendant(1)
            return np.zeros(default_size)
        
        vector = []

        # 1. Structural Features (normalized)
        node_type = node_info.get('type', '')
        has_children = len(node_data.get('children', [])) > 0
        num_children = len(node_data.get('children', []))

        vector.extend([
            hash(node_type) % 1000 / 1000.0,  # Normalize hash
            # hash(tag) % 1000 / 1000.0,
            float(has_children),
            min(num_children / 10000.0, 1.0),  # Normalize to 0-1
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

    def create_parent_feature_vector(self, children_vectors: List[np.ndarray], node_data: Dict[str, Any]) -> np.ndarray:
        """
        Create parent node feature vector by averaging children feature vectors

        Args:
            children_vectors: List of children feature vectors
            node_data: Parent node data

        Returns:
            Feature vector for parent node
        """
        tag = node_data.get('tag', '')
        
        # Check if tag is ICON or SVG - if so, return default vector of ones
        if tag in ["ICON", "SVG"]:
            # Create default vector with ones
            default_size = 17  # structural(3) + style(12) + content(1) + descendant(1)
            return np.zeros(default_size)
        
        if not children_vectors:
            # If no children vectors, create a default vector
            default_size = 0 + 17  # Updated size: structural(3) + style(12) + content(1) + descendant(1)
            return np.zeros(default_size)

        # Average all children feature vectors
        parent_vector = np.mean(children_vectors, axis=0)

        # Override some structural features specific to the parent
        node_info = node_data.get('node', {})
        node_type = node_info.get('type', '')
        num_children = len(children_vectors)

        # Update structural features (first 3 elements)
        parent_vector[0] = hash(node_type) % 1000 / 1000.0
        # parent_vector[1] = hash(tag) % 1000 / 1000.0
        parent_vector[1] = 1.0  # Parent always has children
        parent_vector[2] = min(num_children / 10000.0, 1.0)

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
    
    def check_similarity(self, threshold: float = None) -> Dict[str, List[str]]:
        """
        Check similarity between non-leaf nodes and return groups of similar nodes.
        Leaf nodes are excluded from grouping.
        
        Args:
            threshold: Similarity threshold (uses instance threshold if None)
            
        Returns:
            Dictionary mapping group_id to list of similar node paths (non-leaf nodes only)
        """
        if self.similarity_matrix is None:
            raise ValueError("Must build similarity matrix first")
        
        if len(self.similarity_matrix) == 0:
            logger.warning("No non-leaf nodes available for similarity comparison")
            return {}
        
        threshold = threshold or self.similarity_threshold
        
        # Method 1: Simple threshold-based grouping (only for non-leaf nodes)
        similarity_groups = self._threshold_based_grouping(threshold)
        
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







@dataclass
class NodeSignature:
    """Represents a compact signature of a node for comparison"""
    tag: str
    node_type: str
    has_text: bool
    text_content: str
    style_hash: str
    layout_mode: str
    children_count: int
    children_signatures: List[str]  # Ordered list of child signatures
    
    def __post_init__(self):
        # Normalize text content for comparison
        self.text_content = self.text_content.strip().lower() if self.text_content else ""
        # Create a signature string for this node
        self.signature = self._create_signature()
    
    def _create_signature(self) -> str:
        """Create a unique signature string for this node"""
        children_sig = "|".join(self.children_signatures)
        return f"{self.tag}:{self.node_type}:{self.has_text}:{self.style_hash}:{self.layout_mode}:{self.children_count}:{children_sig}"

class MinEditDistanceSemanticGrouper:
    """
    Semantic grouper using minimum edit distance approach for tree comparison.
    This approach compares the structural similarity of component trees.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 structure_weight: float = 0.6,
                 style_weight: float = 0.3,
                 content_weight: float = 0.1,
                 ignore_text_content: bool = False):
        """
        Initialize the min edit distance grouper
        
        Args:
            similarity_threshold: Threshold for considering nodes similar (0-1)
            structure_weight: Weight for structural similarity
            style_weight: Weight for style similarity  
            content_weight: Weight for content similarity
            ignore_text_content: Whether to ignore text content in comparison
        """
        self.similarity_threshold = similarity_threshold
        self.structure_weight = structure_weight
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.ignore_text_content = ignore_text_content
        
        self.node_signatures = {}  # path -> NodeSignature
        self.node_trees = {}  # path -> tree structure
        self.similarity_matrix = None
        self.non_leaf_paths = []
        
    def extract_node_signature(self, node_data: Dict[str, Any]) -> NodeSignature:
        """Extract signature from a node"""
        node_info = node_data.get('node', {})
        tag = node_data.get('tag', '')
        name = node_data.get('name', '')
        
        # Check if tag is ICON or SVG - if so, return default signature
        if tag in ["ICON", "SVG"]:
            return NodeSignature(
                tag=tag,
                node_type="ICON_SVG",
                has_text=False,
                text_content="",
                style_hash="icon_svg_default",
                layout_mode="NONE",
                children_count=0,
                children_signatures=[]
            )
        
        # Basic properties
        node_type = node_info.get('type', '')
        has_text = bool(node_info.get('characters', ''))
        text_content = node_info.get('characters', '') if not self.ignore_text_content else ""
        layout_mode = node_info.get('layoutMode', 'NONE')
        children_count = len(node_data.get('children', []))
        
        # Style hash (simplified - you can expand this)
        style_hash = self._compute_style_hash(node_info)
        
        # Children signatures will be filled later
        children_signatures = []
        
        return NodeSignature(
            tag=tag,
            node_type=node_type,
            has_text=has_text,
            text_content=text_content,
            style_hash=style_hash,
            layout_mode=layout_mode,
            children_count=children_count,
            children_signatures=children_signatures
        )
    
    def _compute_style_hash(self, node_info: Dict[str, Any]) -> str:
        """Compute a hash representing the style properties of a node"""
        style_props = []
        
        # Text style
        text_style = node_info.get('textStyle', {})
        if text_style:
            style_props.extend([
                text_style.get('fontFamily', ''),
                str(text_style.get('fontSize', 0)),
                text_style.get('fontStyle', ''),
                str(text_style.get('fontWeight', 0))
            ])
        
        # Fill colors
        fills = node_info.get('fills', [])
        if fills:
            fill = fills[0]
            color = fill.get('color', {})
            style_props.extend([
                str(color.get('r', 0)),
                str(color.get('g', 0)),
                str(color.get('b', 0)),
                str(color.get('a', 1))
            ])
        
        # Stroke colors
        strokes = node_info.get('strokes', [])
        if strokes:
            stroke = strokes[0]
            color = stroke.get('color', {})
            style_props.extend([
                str(color.get('r', 0)),
                str(color.get('g', 0)),
                str(color.get('b', 0)),
                str(color.get('a', 1))
            ])
        
        # Layout properties
        style_props.extend([
            str(node_info.get('layoutMode', 'NONE')),
            str(node_info.get('paddingLeft', 0)),
            str(node_info.get('paddingRight', 0)),
            str(node_info.get('paddingTop', 0)),
            str(node_info.get('paddingBottom', 0)),
            str(node_info.get('itemSpacing', 0))
        ])
        
        style_string = "|".join(style_props)
        return hashlib.md5(style_string.encode()).hexdigest()[:8]
    
    def build_node_signatures(self, data: Dict[str, Any], path: str = "") -> None:
        """Build node signatures recursively"""
        if isinstance(data, dict) and 'node' in data:
            current_path = f"{path}/{data.get('name', 'unnamed')}" if path else data.get('name', 'root')
            
            # Extract signature for current node
            signature = self.extract_node_signature(data)
            
            # Check if this is an ICON or SVG node
            tag = data.get('tag', '')
            if tag in ["ICON", "SVG"]:
                # For ICON/SVG nodes, don't process children and store as leaf
                signature.children_signatures = []
                signature.signature = signature._create_signature()
                self.node_signatures[current_path] = signature
                
                # Store tree structure as leaf node
                self.node_trees[current_path] = {
                    'data': data,
                    'children': [],
                    'is_leaf': True,
                    'is_icon_svg': True
                }
                return
            
            # Process children first to get their signatures
            children_signatures = []
            if 'children' in data:
                for child in data['children']:
                    child_path = f"{current_path}"
                    self.build_node_signatures(child, child_path)
                    child_node_path = f"{current_path}/{child.get('name', 'unnamed')}"
                    if child_node_path in self.node_signatures:
                        children_signatures.append(self.node_signatures[child_node_path].signature)
            
            # Update signature with children signatures
            signature.children_signatures = children_signatures
            signature.signature = signature._create_signature()
            
            self.node_signatures[current_path] = signature
            
            # Store tree structure
            self.node_trees[current_path] = {
                'data': data,
                'children': [f"{current_path}/{child.get('name', 'unnamed')}" for child in data.get('children', [])],
                'is_leaf': len(data.get('children', [])) == 0,
                'is_icon_svg': False
            }
    
    def tree_edit_distance(self, tree1_path: str, tree2_path: str) -> float:
        """
        Calculate tree edit distance between two trees
        Returns normalized distance (0 = identical, 1 = completely different)
        """
        if tree1_path not in self.node_signatures or tree2_path not in self.node_signatures:
            return 1.0
        
        sig1 = self.node_signatures[tree1_path]
        sig2 = self.node_signatures[tree2_path]
        
        # Special handling for ICON/SVG nodes
        tree1_info = self.node_trees.get(tree1_path, {})
        tree2_info = self.node_trees.get(tree2_path, {})
        
        # If both are ICON/SVG nodes, they are considered identical
        if (tree1_info.get('is_icon_svg', False) and tree2_info.get('is_icon_svg', False)):
            return 0.0
        
        # If only one is ICON/SVG, they are considered very different
        if (tree1_info.get('is_icon_svg', False) or tree2_info.get('is_icon_svg', False)):
            return 1.0
        
        # Use dynamic programming to calculate edit distance
        return self._tree_edit_distance_dp(sig1, sig2)
    
    def _tree_edit_distance_dp(self, sig1: NodeSignature, sig2: NodeSignature) -> float:
        """Dynamic programming implementation of tree edit distance"""
        
        # Node-level similarity
        node_similarity = self._node_similarity(sig1, sig2)
        
        # If nodes are very different, return high distance
        if node_similarity < 0.1:
            return 1.0
        
        # Structure similarity based on children
        children1 = sig1.children_signatures
        children2 = sig2.children_signatures
        
        if not children1 and not children2:
            # Both are leaf nodes, return distance based on node similarity
            return 1.0 - node_similarity
        
        if not children1 or not children2:
            # One is leaf, other is not
            structure_penalty = 0.5
            return 1.0 - (node_similarity * (1.0 - structure_penalty))
        
        # Both have children - calculate edit distance on children sequences
        children_distance = self._sequence_edit_distance(children1, children2)
        
        # Combine node similarity and children structure similarity
        total_similarity = (
            self.structure_weight * (1.0 - children_distance) +
            (self.style_weight + self.content_weight) * node_similarity
        )
        
        return 1.0 - total_similarity
    
    def _node_similarity(self, sig1: NodeSignature, sig2: NodeSignature) -> float:
        """Calculate similarity between two node signatures"""
        # Special handling for ICON/SVG nodes
        if sig1.tag in ["ICON", "SVG"] and sig2.tag in ["ICON", "SVG"]:
            return 1.0  # All ICON/SVG nodes are considered similar
        
        if sig1.tag in ["ICON", "SVG"] or sig2.tag in ["ICON", "SVG"]:
            return 0.0  # ICON/SVG nodes are not similar to other node types
        
        similarities = []
        
        # Tag and type similarity
        tag_sim = 1.0 if sig1.tag == sig2.tag else 0.0
        type_sim = 1.0 if sig1.node_type == sig2.node_type else 0.0
        
        # Layout similarity
        layout_sim = 1.0 if sig1.layout_mode == sig2.layout_mode else 0.0
        
        # Style similarity
        style_sim = 1.0 if sig1.style_hash == sig2.style_hash else 0.0
        
        # Content similarity
        content_sim = 1.0
        if not self.ignore_text_content:
            if sig1.has_text and sig2.has_text:
                content_sim = self._text_similarity(sig1.text_content, sig2.text_content)
            elif sig1.has_text != sig2.has_text:
                content_sim = 0.0
        
        # Children count similarity
        max_children = max(sig1.children_count, sig2.children_count)
        children_count_sim = 1.0 - abs(sig1.children_count - sig2.children_count) / max(max_children, 1)
        
        # Weighted combination
        total_similarity = (
            self.structure_weight * (tag_sim * 0.4 + type_sim * 0.3 + layout_sim * 0.3) +
            self.style_weight * style_sim +
            self.content_weight * content_sim
        ) * children_count_sim
        
        return total_similarity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple metrics"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Length similarity
        max_len = max(len(text1), len(text2))
        len_sim = 1.0 - abs(len(text1) - len(text2)) / max_len
        
        # Common characters ratio
        common_chars = len(set(text1) & set(text2))
        total_chars = len(set(text1) | set(text2))
        char_sim = common_chars / max(total_chars, 1)
        
        return (len_sim + char_sim) / 2.0
    
    def _sequence_edit_distance(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate normalized edit distance between two sequences"""
        if not seq1 and not seq2:
            return 0.0
        
        if not seq1 or not seq2:
            return 1.0
        
        # Dynamic programming for edit distance
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Normalize by maximum possible distance
        max_distance = max(m, n)
        return dp[m][n] / max_distance if max_distance > 0 else 0.0
    
    def build_similarity_matrix(self, nodes_data: Dict[str, Any]) -> np.ndarray:
        """Build similarity matrix using min edit distance approach"""
        
        # Step 1: Build node signatures
        self.build_node_signatures(nodes_data)
        
        # Step 2: Filter non-leaf nodes for comparison (excluding ICON/SVG nodes)
        self.non_leaf_paths = [
            path for path, tree_info in self.node_trees.items() 
            if not tree_info['is_leaf'] and not tree_info.get('is_icon_svg', False)
        ]
        
        if not self.non_leaf_paths:
            logger.warning("No non-leaf nodes found for similarity comparison")
            return np.array([])
        
        # Step 3: Calculate similarity matrix
        n = len(self.non_leaf_paths)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Calculate edit distance and convert to similarity
                    edit_distance = self.tree_edit_distance(
                        self.non_leaf_paths[i], 
                        self.non_leaf_paths[j]
                    )
                    similarity_matrix[i][j] = 1.0 - edit_distance
        
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def check_similarity(self, threshold: float = None) -> Dict[str, List[str]]:
        """Find groups of similar nodes based on edit distance"""
        if self.similarity_matrix is None:
            raise ValueError("Must build similarity matrix first")
        
        if len(self.similarity_matrix) == 0:
            logger.warning("No non-leaf nodes available for similarity comparison")
            return {}
        
        threshold = threshold or self.similarity_threshold
        
        # Use connected components to find similarity groups
        return self._find_similarity_groups(threshold)
    
    def _find_similarity_groups(self, threshold: float) -> Dict[str, List[str]]:
        """Find connected components of similar nodes"""
        n = len(self.non_leaf_paths)
        visited = [False] * n
        groups = {}
        group_counter = 0
        
        def dfs(node_idx: int, group: List[int]):
            visited[node_idx] = True
            group.append(node_idx)
            
            for j in range(n):
                if not visited[j] and self.similarity_matrix[node_idx][j] >= threshold:
                    dfs(j, group)
        
        for i in range(n):
            if not visited[i]:
                group = []
                dfs(i, group)
                
                if len(group) > 1:  # Only create groups with multiple nodes
                    group_id = f"group_{group_counter}"
                    groups[group_id] = [self.non_leaf_paths[idx] for idx in group]
                    group_counter += 1
        
        return groups
    
    def add_node_ids_to_json(self, original_data: Dict[str, Any], similarity_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Add node_id to each node based on similarity groups"""
        # Create mapping from node path to group id
        path_to_group = {}
        for group_id, node_paths in similarity_groups.items():
            for node_path in node_paths:
                path_to_group[node_path] = group_id
        
        # Add node_ids recursively
        return self._add_node_ids_recursive(original_data, path_to_group)
    
    def _add_node_ids_recursive(self, data: Dict[str, Any], path_to_group: Dict[str, str], path: str = "") -> Dict[str, Any]:
        """Recursively add node_ids to the JSON structure"""
        if isinstance(data, dict):
            result = data.copy()
            
            if 'node' in data:  # This is a node
                current_path = f"{path}/{data.get('name', 'unnamed')}" if path else data.get('name', 'root')
                
                # Check if this is a leaf node or ICON/SVG node
                tag = data.get('tag', '')
                is_leaf = len(data.get('children', [])) == 0
                is_icon_svg = tag in ["ICON", "SVG"]
                
                if is_leaf or is_icon_svg:
                    # Leaf nodes and ICON/SVG nodes get unique IDs
                    if is_icon_svg:
                        result['node_id'] = f"icon_svg_{hashlib.md5(current_path.encode()).hexdigest()[:8]}"
                    else:
                        result['node_id'] = f"leaf_{hashlib.md5(current_path.encode()).hexdigest()[:8]}"
                else:
                    # Non-leaf nodes get group IDs if they're in a similarity group
                    if current_path in path_to_group:
                        result['node_id'] = path_to_group[current_path]
                    else:
                        # Generate unique ID for non-grouped non-leaf nodes
                        result['node_id'] = f"unique_{hashlib.md5(current_path.encode()).hexdigest()[:8]}"
            
            if 'children' in data and not data.get('tag', '') in ["ICON", "SVG"]:
                # Don't process children for ICON/SVG nodes
                child_path = f"{path}/{data.get('name', 'unnamed')}" if path else data.get('name', 'root')
                result['children'] = [
                    self._add_node_ids_recursive(child, path_to_group, child_path)
                    for child in data['children']
                ]
            
            return result
        
        return data
    
    def get_analysis_info(self) -> Dict[str, Any]:
        """Get analysis information for debugging"""
        if not self.node_signatures:
            return {"error": "No analysis performed yet"}
        
        leaf_count = sum(1 for tree_info in self.node_trees.values() if tree_info['is_leaf'])
        non_leaf_count = len(self.node_trees) - leaf_count
        icon_svg_count = sum(1 for tree_info in self.node_trees.values() if tree_info.get('is_icon_svg', False))
        
        return {
            "total_nodes": len(self.node_signatures),
            "leaf_nodes": leaf_count,
            "non_leaf_nodes": non_leaf_count,
            "icon_svg_nodes": icon_svg_count,
            "comparison_approach": "minimum_edit_distance",
            "weights": {
                "structure": self.structure_weight,
                "style": self.style_weight,
                "content": self.content_weight
            },
            "ignore_text_content": self.ignore_text_content,
            "icon_svg_handling": "treated_as_identical_leaf_nodes"
        }





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

def add_node_ids(data):
    """
    Add unique node_id to any node that doesn't have one
    """
    def process_node(node):
        if isinstance(node, dict):
            # If it's a node-like object and missing node_id, add one
            if ('node' in node or 'name' in node) and 'node_id' not in node:
                node['node_id'] = f"unique_{uuid.uuid4().hex[:8]}" # Short unique ID
            
            # Process all values recursively
            for value in node.values():
                if isinstance(value, (dict, list)):
                    process_node(value)
        
        elif isinstance(node, list):
            for item in node:
                process_node(item)
    
    process_node(data)
    return data

# JSON Processing Functions
def generate_unique_id():
    """Generate a unique ID similar to the existing format"""
    return f"unique_{uuid.uuid4().hex[:8]}"

def is_group_node(node_id):
    """Check if node_id matches the group_X pattern"""
    return bool(re.match(r'^group_\d+$', node_id))

def convert_children_ids(node, visited=None):
    """Recursively convert all children node_ids to unique IDs"""
    if visited is None:
        visited = set()
    
    # Avoid infinite recursion
    if id(node) in visited:
        return
    visited.add(id(node))
    
    # Process children if they exist
    if 'children' in node and isinstance(node['children'], list):
        for child in node['children']:
            # Convert child's node_id if it exists
            if 'node_id' in child:
                child['node_id'] = generate_unique_id()
            
            # Recursively process the child's children
            convert_children_ids(child, visited)

def process_json_tree(data):
    """Process the entire JSON tree looking for group nodes"""
    def traverse(node):
        # Check if current node has node_id with group_X pattern
        if isinstance(node, dict) and 'node_id' in node and is_group_node(node['node_id']):
            print(f"Found group node: {node['node_id']}")
            print("Converting all children node_ids to unique IDs...")
            convert_children_ids(node)
        
        # Recursively traverse children
        if isinstance(node, dict) and 'children' in node:
            for child in node['children']:
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(data)
    return data

#################################################### UTILS ####################################################
###############################################################################################################
###############################################################################################################

def visualize_groups_on_image(image_path: str, json_data: Dict[str, Any], output_dir: str = "./output"):
    """
    Simpler version that works with PNG/JPG images directly.
    
    Args:
        image_path: Path to the image file (PNG, JPG, etc.)
        json_data: JSON data containing nodes with coordinates and group_ids
        output_dir: Directory to save output images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base image
    base_image = Image.open(image_path)
    
    # Extract node coordinates and group information
    node_groups = {}
    
    def extract_nodes_recursive(node, path=""):
        if isinstance(node, dict):
            if 'node' in node:
                current_path = f"{path}/{node.get('name', 'unnamed')}" if path else node.get('name', 'root')
                
                # Get coordinates from absoluteBoundingBox
                node_info = node.get('node', {})
                x = node_info.get('x', 0)
                y = node_info.get('y', 0)
                width = node_info.get('width', 0)
                height = node_info.get('height', 0)
                
                group_id = node.get('node_id', 'no_group')
                
                if group_id not in node_groups:
                    node_groups[group_id] = []
                
                node_groups[group_id].append({
                    'name': node.get('name', 'unnamed'),
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })
                
                # Process children
                for child in node.get('children', []):
                    extract_nodes_recursive(child, current_path)
    
    extract_nodes_recursive(json_data)
    
    # Generate colors for each group
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]
    
    color_index = 0
    group_colors = {}
    
    # Create image for each group
    for group_id, nodes in node_groups.items():
        if len(nodes) <= 1:  # Skip groups with only one node
            continue
            
        # Assign color to group
        if group_id not in group_colors:
            group_colors[group_id] = colors[color_index % len(colors)]
            color_index += 1
        
        # Create a copy of the base image
        group_image = base_image.copy()
        draw = ImageDraw.Draw(group_image)
        
        # Draw rectangles around nodes in this group
        color = group_colors[group_id]
        for node in nodes:
            x, y, width, height = int(node['x']), int(node['y']), int(node['width']), int(node['height'])
            
            # Draw rectangle outline
            draw.rectangle(
                [(x, y), (x + width, y + height)],
                outline=color,
                width=4
            )
            
            # Add group label
            draw.text((x, max(0, y - 25)), f"{group_id}", fill=color)
        
        # Save the image
        output_path = f"{output_dir}/group_{group_id}.png"
        group_image.save(output_path)
        print(f"Saved: {output_path}")
    
    print(f"Total groups found: {len(node_groups)}")
    print(f"Created {len([g for g in node_groups.values() if len(g) > 1])} group visualization images")



# with open('output.json', 'r') as f:
#     json_data = json.load(f)

# # For PNG/JPG images (simpler):
# visualize_groups_on_image('PAGE_109.png', json_data)


def print_figma_node(node, depth=0):
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

    # Print current node info
    print(f"{indent}- {display_name} [{tag}] -> {name} {layout_str} ({node_id})")

    # Recursively print children
    for child in node.get("children", []):
        print_figma_node(child, depth + 1)



# with open("restored.json", "r") as f:
#     restored_data = json.load(f)
# print_figma_node(restored_data)


###############################################################################################################
###############################################################################################################
###############################################################################################################




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


# Alternative endpoint if you want to return just the processed data
@app.post("/api")
async def process_figma_data_simple(json_data: JSONData) -> Dict[str, Any]:
    """
    Simplified version that returns only the processed JSON data
    """
    try:
        import copy
        data = copy.deepcopy(json_data.data)
        
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
        similarity_groups = detector.check_similarity()
        result_json = detector.add_node_ids_to_json(data, similarity_groups)
        
        # Restore original names
        restore_names_from_mapping(result_json, name_mapping)
        
        return result_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/min-edit-distance")
async def process_figma_data_min_edit(json_data: JSONData) -> Dict[str, Any]:
    """
    Process Figma JSON data using Minimum Edit Distance Tree approach
    """
    try:
        import copy
        data = copy.deepcopy(json_data.data)
        
        # Rename names and create mapping
        name_mapping = {}
        counter = [0]
        rename_names_and_store_mapping(data, counter, name_mapping)
        
        # Initialize the min edit distance grouper
        grouper = MinEditDistanceSemanticGrouper(
            similarity_threshold=0.99999,  # Higher threshold for more precise grouping
            structure_weight=0.6,
            style_weight=0.3,
            content_weight=0.1,
            ignore_text_content=False
        )
        
        # Build similarity matrix
        similarity_matrix = grouper.build_similarity_matrix(data)
        
        # Find similar groups
        similarity_groups = grouper.check_similarity()
        
        # Add node_ids to JSON
        result_json = grouper.add_node_ids_to_json(data, similarity_groups)
        
        # Restore original names
        restore_names_from_mapping(result_json, name_mapping)

        result_json = add_node_ids(result_json)

        
        return result_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Min edit distance processing error: {str(e)}")
    


# New Enhanced Endpoints with JSON Processing
@app.post("/api/enhanced")
async def process_figma_data_simple_enhanced(json_data: JSONData) -> Dict[str, Any]:
    """
    Enhanced version that applies JSON tree processing before returning the result
    """
    try:
        data = copy.deepcopy(json_data.data)
        
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
        similarity_groups = detector.check_similarity()
        result_json = detector.add_node_ids_to_json(data, similarity_groups)
        
        # Restore original names
        restore_names_from_mapping(result_json, name_mapping)
        
        # Apply JSON tree processing before returning
        processed_result = process_json_tree(copy.deepcopy(result_json))
        
        return processed_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced processing error: {str(e)}")

@app.post("/api/min-edit-distance/enhanced")
async def process_figma_data_min_edit_enhanced(json_data: JSONData) -> Dict[str, Any]:
    """
    Enhanced version using Minimum Edit Distance Tree approach with JSON processing
    """
    try:
        data = copy.deepcopy(json_data.data)
        
        # Rename names and create mapping
        name_mapping = {}
        counter = [0]
        rename_names_and_store_mapping(data, counter, name_mapping)
        
        # Initialize the min edit distance grouper
        grouper = MinEditDistanceSemanticGrouper(
            similarity_threshold=0.99999,  # Higher threshold for more precise grouping
            structure_weight=0.6,
            style_weight=0.3,
            content_weight=0.1,
            ignore_text_content=False
        )
        
        # Build similarity matrix
        similarity_matrix = grouper.build_similarity_matrix(data)
        
        # Find similar groups
        similarity_groups = grouper.check_similarity()
        
        # Add node_ids to JSON
        result_json = grouper.add_node_ids_to_json(data, similarity_groups)
        
        # Restore original names
        restore_names_from_mapping(result_json, name_mapping)
        
        # Apply JSON tree processing before returning
        processed_result = process_json_tree(copy.deepcopy(result_json))

        processed_result = add_node_ids(processed_result)

        
        return processed_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced min edit distance processing error: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)