import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from SpatialQuery.spatial_query import spatial_query
from SpatialQuery.spatial_query_multiple_fov import spatial_query_multi
import anndata as ad

app = Flask(__name__)

# Enable CORS for specific ports
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5001",                     # local dev
            "http://localhost:6006",                     # local dev
            # "https://scfind.dev.hubmapconsortium.org",   # staging/prod
            # "https://scfind.hubmapconsortium.org",        # official prod
            # "https://portal.dev.hubmapconsortium.org",     # portal dev
            # "https://portal.test.hubmapconsortium.org",     # portal test
            # "https://portal-prod.test.hubmapconsortium.org", # portal prod
            # "https://portal.hubmapconsortium.org" # portal prod
        ]
    }
}, supports_credentials=True)

spatial_query_objects = {}
# PROCESSED_FOLDER = "/tmp/spatial_objs"
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# save in memory for now, can save to disk or S3 later
@app.route('/api/upload_h5ad', methods=['POST'])
def upload_h5ad():
    file = request.files.get('file')
    dataset_id = request.form.get('dataset_id')

    if not file:
        return jsonify({"error": "Missing file"}), 400
    if not dataset_id:
        return jsonify({"error": "Missing dataset_id"}), 400

    if dataset_id in spatial_query_objects:
        return jsonify({
            "message": f"Dataset '{dataset_id}' already exists. Skipping upload.",
            "dataset_id": dataset_id,
            "skipped": True
        })

    try:
        adata = ad.read_h5ad(file)
        spatial_obj = spatial_query(adata=adata)
        spatial_query_objects[dataset_id] = spatial_obj

        return jsonify({
            "message": f"Dataset '{dataset_id}' uploaded successfully.",
            "dataset_id": dataset_id,
            "skipped": False
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/find_fp_knn', methods=['GET', 'POST'])
def find_fp_knn_api():
    """
    Find frequent patterns within the K-nearest-neighbors of certain cell type.
    
    Parameters:
    - ct: str - Center cell type of interest
    - k: int - Number of nearest neighbors
    - min_support: float - Threshold of frequency to consider a pattern as a frequent pattern
    - dataset_id: str - Dataset identifier for the spatial query object
    
    Returns:
    - A data frame containing frequent patterns in the neighborhood of certain cell type along with frequencies
    """
    try:
        if request.method == 'POST':
            data = request.get_json()
            ct = data.get('ct')
            k = data.get('k', 30)
            min_support = data.get('min_support', 0.5)
            dataset_id = data.get('dataset_id')
        else:  # GET
            ct = request.args.get('ct')
            k = request.args.get('k', 30, type=int)
            min_support = request.args.get('min_support', 0.5, type=float)
            dataset_id = request.args.get('dataset_id')

        if not ct:
            return jsonify({"error": "Missing 'ct' parameter"}), 400
        
        if not dataset_id or dataset_id not in spatial_query_objects:
            return jsonify({"error": "Invalid or missing 'dataset_id' parameter"}), 400

        spatial_obj = spatial_query_objects[dataset_id]
        result = spatial_obj.find_fp_knn(ct=ct, k=k, min_support=min_support)
        
        if isinstance(result, pd.DataFrame):
            return jsonify({"frequent_patterns": result.to_dict(orient='records')})
        else:
            return jsonify({"frequent_patterns": result})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/find_fp_dist', methods=['GET', 'POST'])
def find_fp_dist_api():
    """
    Find frequent patterns within the radius of certain cell type.
    
    Parameters:
    - ct: str - Center cell type of interest
    - max_dist: float - The radius within which cells are considered neighbors of the center cell type
    - min_size: int - Minimum number of neighboring cells required to keep a neighborhood (default: 0)
    - min_support: float - Threshold of frequency to consider a pattern as a frequent pattern
    - dataset_id: str - Dataset identifier for the spatial query object
    
    Returns:
    - A data frame containing frequent patterns in the neighborhood of certain cell type along with frequencies
    """
    try:
        if request.method == 'POST':
            data = request.get_json()
            ct = data.get('ct')
            max_dist = data.get('max_dist', 100.0)
            min_size = data.get('min_size', 0)
            min_support = data.get('min_support', 0.5)
            dataset_id = data.get('dataset_id')
        else:  # GET
            ct = request.args.get('ct')
            max_dist = request.args.get('max_dist', 100.0, type=float)
            min_size = request.args.get('min_size', 0, type=int)
            min_support = request.args.get('min_support', 0.5, type=float)
            dataset_id = request.args.get('dataset_id')

        if not ct:
            return jsonify({"error": "Missing 'ct' parameter"}), 400
        
        if not dataset_id or dataset_id not in spatial_query_objects:
            return jsonify({"error": "Invalid or missing 'dataset_id' parameter"}), 400

        spatial_obj = spatial_query_objects[dataset_id]
        result = spatial_obj.find_fp_dist(ct=ct, max_dist=max_dist, min_size=min_size, min_support=min_support)
        
        if isinstance(result, pd.DataFrame):
            return jsonify({"frequent_patterns": result.to_dict(orient='records')})
        else:
            return jsonify({"frequent_patterns": result})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/motif_enrichment_knn', methods=['GET', 'POST'])
def motif_enrichment_knn_api():
    """
    Perform motif (pattern) enrichment analysis using k-nearest-neighbors (KNN) with hypergeometric test.
    
    Parameters:
    - ct: str - The cell type of the center cell
    - motifs: str or list of str - Specified motifs to be tested (optional)
    - k: int - Number of nearest neighbors to consider
    - min_support: float - Threshold of frequency to consider a pattern as a frequent pattern
    - max_dist: float - Maximum distance for neighbors (default: 200)
    - return_cellID: bool - Whether to return cell indices (default: False)
    - dataset_id: str - Dataset identifier for the spatial query object
    
    Returns:
    - A data frame containing enrichment analysis results
    """
    try:
        if request.method == 'POST':
            data = request.get_json()
            ct = data.get('ct')
            motifs = data.get('motifs')
            k = data.get('k', 30)
            min_support = data.get('min_support', 0.5)
            max_dist = data.get('max_dist', 200.0)
            return_cellID = data.get('return_cellID', False)
            dataset_id = data.get('dataset_id')
        else:  # GET
            ct = request.args.get('ct')
            motifs = request.args.get('motifs')
            k = request.args.get('k', 30, type=int)
            min_support = request.args.get('min_support', 0.5, type=float)
            max_dist = request.args.get('max_dist', 200.0, type=float)
            return_cellID = request.args.get('return_cellID', 'false').lower() == 'true'
            dataset_id = request.args.get('dataset_id')

        if not ct:
            return jsonify({"error": "Missing 'ct' parameter"}), 400
        
        if not dataset_id or dataset_id not in spatial_query_objects:
            return jsonify({"error": "Invalid or missing 'dataset_id' parameter"}), 400

        if motifs and isinstance(motifs, str):
            motifs = [motif.strip() for motif in motifs.split(',')]

        spatial_obj = spatial_query_objects[dataset_id]
        result = spatial_obj.motif_enrichment_knn(
            ct=ct, 
            motifs=motifs, 
            k=k, 
            min_support=min_support, 
            max_dist=max_dist, 
            return_cellID=return_cellID
        )
        
        if isinstance(result, pd.DataFrame):
            return jsonify({"enrichment_results": result.to_dict(orient='records')})
        else:
            return jsonify({"enrichment_results": result})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/motif_enrichment_dist', methods=['GET', 'POST'])
def motif_enrichment_dist_api():
    """
    Perform motif (pattern) enrichment analysis for radius-based neighborhoods with hypergeometric test.
    
    Parameters:
    - ct: str - The cell type of the center cell
    - motifs: str or list of str - Specified motifs to be tested (optional)
    - max_dist: float - The radius within which cells are considered neighbors of the center cell type
    - min_size: int - Minimum number of neighboring cells required to keep a neighborhood (default: 0)
    - min_support: float - Threshold of frequency to consider a pattern as a frequent pattern
    - max_ns: int - Maximum number of neighborhood size for each point (default: 100)
    - return_cellID: bool - Whether to return cell indices (default: False)
    - dataset_id: str - Dataset identifier for the spatial query object
    
    Returns:
    - A data frame containing enrichment analysis results
    """
    try:
        if request.method == 'POST':
            data = request.get_json()
            ct = data.get('ct')
            motifs = data.get('motifs')
            max_dist = data.get('max_dist', 100.0)
            min_size = data.get('min_size', 0)
            min_support = data.get('min_support', 0.5)
            max_ns = data.get('max_ns', 100)
            return_cellID = data.get('return_cellID', False)
            dataset_id = data.get('dataset_id')
        else:  # GET
            ct = request.args.get('ct')
            motifs = request.args.get('motifs')
            max_dist = request.args.get('max_dist', 100.0, type=float)
            min_size = request.args.get('min_size', 0, type=int)
            min_support = request.args.get('min_support', 0.5, type=float)
            max_ns = request.args.get('max_ns', 100, type=int)
            return_cellID = request.args.get('return_cellID', 'false').lower() == 'true'
            dataset_id = request.args.get('dataset_id')

        if not ct:
            return jsonify({"error": "Missing 'ct' parameter"}), 400
        
        if not dataset_id or dataset_id not in spatial_query_objects:
            return jsonify({"error": "Invalid or missing 'dataset_id' parameter"}), 400

        # Convert motifs string to list if provided
        if motifs and isinstance(motifs, str):
            motifs = [motif.strip() for motif in motifs.split(',')]

        spatial_obj = spatial_query_objects[dataset_id]
        result = spatial_obj.motif_enrichment_dist(
            ct=ct, 
            motifs=motifs, 
            max_dist=max_dist, 
            min_size=min_size, 
            min_support=min_support, 
            max_ns=max_ns, 
            return_cellID=return_cellID
        )
        
        if isinstance(result, pd.DataFrame):
            return jsonify({"enrichment_results": result.to_dict(orient='records')})
        else:
            return jsonify({"enrichment_results": result})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/load_spatial_data', methods=['POST'])
def load_spatial_data_api():
    """
    Load spatial data and create a spatial query object.
    
    Parameters:
    - dataset_id: str - Unique identifier for the dataset
    - data: dict - AnnData object data or file path
    - dataset: str - Dataset name (default: 'ST')
    - spatial_key: str - Spatial coordination key (default: 'X_spatial')
    - label_key: str - Label key (default: 'predicted_label')
    - leaf_size: int - KD-tree leaf size (default: 10)
    - max_radius: float - Maximum radius (default: 500)
    - n_split: int - Number of splits (default: 10)
    
    Returns:
    - Success message with dataset_id
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        ann_data = data.get('data')
        dataset = data.get('dataset', 'ST')
        spatial_key = data.get('spatial_key', 'X_spatial')
        label_key = data.get('label_key', 'predicted_label')
        leaf_size = data.get('leaf_size', 10)
        max_radius = data.get('max_radius', 500.0)
        n_split = data.get('n_split', 10)

        if not dataset_id:
            return jsonify({"error": "Missing 'dataset_id' parameter"}), 400
        
        if not ann_data:
            return jsonify({"error": "Missing 'data' parameter"}), 400

        if isinstance(ann_data, dict):
            adata = ad.AnnData(
                X=ann_data.get('X'),
                obs=ann_data.get('obs'),
                var=ann_data.get('var'),
                obsm=ann_data.get('obsm', {})
            )
        else:
            adata = ad.read(ann_data)

        spatial_obj = spatial_query(
            adata=adata,
            dataset=dataset,
            spatial_key=spatial_key,
            label_key=label_key,
            leaf_size=leaf_size,
            max_radius=max_radius,
            n_split=n_split
        )

        spatial_query_objects[dataset_id] = spatial_obj

        return jsonify({
            "message": "Spatial data loaded successfully",
            "dataset_id": dataset_id,
            "available_datasets": list(spatial_query_objects.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/list_datasets', methods=['GET'])
def list_datasets_api():
    """
    List all loaded spatial datasets.
    
    Returns:
    - List of available dataset IDs
    """
    try:
        return jsonify({
            "available_datasets": list(spatial_query_objects.keys()),
            "count": len(spatial_query_objects)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/remove_dataset', methods=['POST'])
def remove_dataset_api():
    """
    Remove a spatial dataset from memory.
    
    Parameters:
    - dataset_id: str - Dataset identifier to remove
    
    Returns:
    - Success message
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')

        if not dataset_id:
            return jsonify({"error": "Missing 'dataset_id' parameter"}), 400

        if dataset_id not in spatial_query_objects:
            return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

        del spatial_query_objects[dataset_id]

        return jsonify({
            "message": f"Dataset '{dataset_id}' removed successfully",
            "available_datasets": list(spatial_query_objects.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return 'ok'

if __name__ == '__main__':
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    port = int(os.environ.get("PORT", 8080))  
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
