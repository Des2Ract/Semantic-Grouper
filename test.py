import json
import os
from collections import defaultdict
from itertools import combinations
import numpy as np

class GroupEvaluator:
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics for a new evaluation run"""
        self.total_pairs = 0
        self.true_positives = 0  # Correctly identified as same group
        self.false_positives = 0  # Incorrectly identified as same group
        self.true_negatives = 0  # Correctly identified as different groups
        self.false_negatives = 0  # Incorrectly identified as different groups
        
        self.pages_processed = 0
        self.total_gt_groups = 0
        self.total_pred_groups = 0
    
    def extract_groups_from_json(self, json_data):
        """
        Extract groups from JSON data
        Returns: dict with group_id as key and list of node names as values
        """
        groups = defaultdict(list)
        
        def traverse_json(node, path=""):
            if isinstance(node, dict):
                # Check if this node has a node_id that indicates it's a group
                node_id = node.get('node_id', '')
                name = node.get('name', f'node_{len(path)}')
                
                if node_id.startswith('group_'):
                    groups[node_id].append(name)
                
                # Recursively traverse children
                if 'children' in node:
                    for child in node['children']:
                        traverse_json(child, path + "/" + name)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    traverse_json(item, path + f"[{i}]")
        
        traverse_json(json_data)
        return dict(groups)
    
    
    def extract_groups_from_json_gt(self, json_data):
        """
        Extract groups from JSON data
        Returns: dict with group_id as key and list of node names as values
        """
        groups = defaultdict(list)
        
        def traverse_json(node, path=""):
            if isinstance(node, dict):
                # Check if this node has a node_id that indicates it's a group
                node_id = node.get('ground_truth', '')
                name = node.get('name', f'node_{len(path)}')
                
                if node_id.startswith('group_'):
                    groups[node_id].append(name)
                
                # Recursively traverse children
                if 'children' in node:
                    for child in node['children']:
                        traverse_json(child, path + "/" + name)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    traverse_json(item, path + f"[{i}]")
        
        traverse_json(json_data)
        return dict(groups)
    
    def get_all_elements(self, groups_dict):
        """Get all unique elements from all groups"""
        all_elements = set()
        for group_elements in groups_dict.values():
            all_elements.update(group_elements)
        return all_elements
    
    def create_element_to_group_mapping(self, groups_dict):
        """Create mapping from element to its group ID"""
        element_to_group = {}
        for group_id, elements in groups_dict.items():
            for element in elements:
                element_to_group[element] = group_id
        return element_to_group
    
    def evaluate_page(self, gt_json, pred_json):
        """
        Evaluate a single page by comparing ground truth and predicted groups
        """
        # Extract groups from both JSONs
        gt_groups = self.extract_groups_from_json_gt(gt_json)
        pred_groups = self.extract_groups_from_json(pred_json)
        
        # Get all elements that appear in either ground truth or predictions
        gt_elements = self.get_all_elements(gt_groups)
        pred_elements = self.get_all_elements(pred_groups)
        all_elements = gt_elements.union(pred_elements)
        
        # Create mappings from element to group
        gt_element_to_group = self.create_element_to_group_mapping(gt_groups)
        pred_element_to_group = self.create_element_to_group_mapping(pred_groups)
        
        # Convert to list for pair generation
        all_elements_list = list(all_elements)
        
        # Generate all pairs and evaluate
        pairs_in_page = 0
        tp_page = fp_page = tn_page = fn_page = 0
        
        for i, j in combinations(range(len(all_elements_list)), 2):
            elem1, elem2 = all_elements_list[i], all_elements_list[j]
            pairs_in_page += 1
            
            # Check if elements are in same group in ground truth
            gt_same_group = (elem1 in gt_element_to_group and 
                           elem2 in gt_element_to_group and 
                           gt_element_to_group[elem1] == gt_element_to_group[elem2])
            
            # Check if elements are in same group in predictions
            pred_same_group = (elem1 in pred_element_to_group and 
                             elem2 in pred_element_to_group and 
                             pred_element_to_group[elem1] == pred_element_to_group[elem2])
            
            # Update confusion matrix
            if gt_same_group and pred_same_group:
                tp_page += 1
            elif not gt_same_group and pred_same_group:
                fp_page += 1
            elif not gt_same_group and not pred_same_group:
                tn_page += 1
            else:  # gt_same_group and not pred_same_group
                fn_page += 1
        
        # Update global metrics
        self.total_pairs += pairs_in_page
        self.true_positives += tp_page
        self.false_positives += fp_page
        self.true_negatives += tn_page
        self.false_negatives += fn_page
        
        self.pages_processed += 1
        self.total_gt_groups += len(gt_groups)
        self.total_pred_groups += len(pred_groups)
        
        return {
            'page_pairs': pairs_in_page,
            'tp': tp_page,
            'fp': fp_page,
            'tn': tn_page,
            'fn': fn_page,
            'gt_groups': len(gt_groups),
            'pred_groups': len(pred_groups)
        }
    
    def calculate_metrics(self):
        """Calculate final evaluation metrics"""
        if self.total_pairs == 0:
            return {}
        
        # Basic metrics
        accuracy = (self.true_positives + self.true_negatives) / self.total_pairs
        
        # Precision, Recall, F1 for group detection
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Group-level metrics
        avg_gt_groups_per_page = self.total_gt_groups / self.pages_processed if self.pages_processed > 0 else 0
        avg_pred_groups_per_page = self.total_pred_groups / self.pages_processed if self.pages_processed > 0 else 0
        
        return {
            'pages_processed': self.pages_processed,
            'total_pairs': self.total_pairs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'avg_gt_groups_per_page': avg_gt_groups_per_page,
            'avg_pred_groups_per_page': avg_pred_groups_per_page,
            'total_gt_groups': self.total_gt_groups,
            'total_pred_groups': self.total_pred_groups
        }
    
    def print_metrics(self):
        """Print formatted metrics"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("No data processed yet!")
            return
        
        print("=" * 60)
        print("GROUP EVALUATION METRICS")
        print("=" * 60)
        print(f"Pages Processed: {metrics['pages_processed']}")
        print(f"Total Element Pairs: {metrics['total_pairs']:,}")
        print()
        print("CONFUSION MATRIX:")
        print(f"  True Positives:  {metrics['true_positives']:,}")
        print(f"  False Positives: {metrics['false_positives']:,}")
        print(f"  True Negatives:  {metrics['true_negatives']:,}")
        print(f"  False Negatives: {metrics['false_negatives']:,}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print()
        print("GROUP STATISTICS:")
        print(f"  Total GT Groups: {metrics['total_gt_groups']}")
        print(f"  Total Predicted Groups: {metrics['total_pred_groups']}")
        print(f"  Avg GT Groups per Page: {metrics['avg_gt_groups_per_page']:.2f}")
        print(f"  Avg Predicted Groups per Page: {metrics['avg_pred_groups_per_page']:.2f}")
        print("=" * 60)


def evaluate_from_files(gt_file_path, pred_file_path):
    """
    Evaluate a single pair of ground truth and prediction files
    """
    evaluator = GroupEvaluator()
    
    try:
        # Load ground truth JSON
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            gt_json = json.load(f)
        
        # Load prediction JSON
        with open(pred_file_path, 'r', encoding='utf-8') as f:
            pred_json = json.load(f)
        
        # Evaluate
        page_results = evaluator.evaluate_page(gt_json, pred_json)
        
        print(f"Evaluated: {gt_file_path} vs {pred_file_path}")
        print(f"Page results: {page_results}")
        
        return evaluator
    
    except Exception as e:
        print(f"Error processing {gt_file_path} and {pred_file_path}: {str(e)}")
        return None


def evaluate_from_directory(directory_path, gt_suffix="_gt.json", pred_suffix="_pred.json"):
    """
    Evaluate all JSON file pairs in a directory
    Assumes files are named like: page1_gt.json, page1_pred.json
    """
    evaluator = GroupEvaluator()
    
    # Find all ground truth files
    gt_files = [f for f in os.listdir(directory_path) if f.endswith(gt_suffix)]
    
    processed_count = 0
    for gt_file in gt_files:
        # Construct corresponding prediction file name
        base_name = gt_file.replace(gt_suffix, "")
        pred_file = base_name + pred_suffix
        
        gt_path = os.path.join(directory_path, gt_file)
        pred_path = os.path.join(directory_path, pred_file)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found for {gt_file}")
            continue
        
        try:
            # Load JSONs
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_json = json.load(f)
            
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_json = json.load(f)
            
            # Evaluate
            page_results = evaluator.evaluate_page(gt_json, pred_json)
            processed_count += 1
            
            print(f"Processed {processed_count}: {gt_file} vs {pred_file}")
            
        except Exception as e:
            print(f"Error processing {gt_file} and {pred_file}: {str(e)}")
            continue
    
    return evaluator


# Example usage
if __name__ == "__main__":
    # Example 1: Evaluate single file pair
    # evaluator = evaluate_from_files("page1_gt.json", "page1_pred.json")
    # if evaluator:
    #     evaluator.print_metrics()
    
    # Example 2: Evaluate all files in a directory
    # evaluator = evaluate_from_directory("./json_files/")
    # evaluator.print_metrics()
    
    # Example 3: Manual evaluation with your provided JSON
    evaluator = GroupEvaluator()
       # Evaluate the example
    evaluator = evaluate_from_directory("./test_data/")
    evaluator.print_metrics()

    # print("\nTo use this script with your files:")
    # print("1. For single file pair:")
    # print('   evaluator = evaluate_from_files("gt_file.json", "pred_file.json")')
    # print("2. For directory of files:")
    # print('   evaluator = evaluate_from_directory("./your_directory/")')
    # print("3. Then call: evaluator.print_metrics()")