# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import json
from typing import List
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Damage Assessment API",
    description="AI-powered vehicle condition assessment for rental businesses",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
try:
    model = YOLO('models/best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

# Damage type to cost mapping (you can customize this)
# Enhanced damage cost mapping with better vehicle part classification
# Enhanced damage cost mapping with better vehicle part classification
DAMAGE_COST_MAP = {
    # Vehicle parts with realistic repair costs
    'bonnet': 400,      # Hood
    'hood': 800,        # Alternative name
    'fender': 300,      # Front fender
    'door': 350,        # Car door
    'bumper': 400,      # Front/rear bumper
    'bumper_front': 500,
    'bumper_rear': 500,
    'windshield': 400,  # Windshield
    'headlight': 300,   # Headlight assembly
    'taillight': 250,   # Taillight
    'mirror': 200,      # Side mirror
    'quarter_panel': 900, # Quarter panel
    'roof': 1000,       # Roof damage
    'trunk': 600,       # Trunk lid
}

# Damage severity multipliers based on confidence
def get_damage_severity_multiplier(confidence: float) -> float:
    """Convert confidence to severity multiplier"""
    if confidence > 0.8:
        return 1.0  # High confidence - full cost
    elif confidence > 0.5:
        return 0.7  # Medium confidence - reduced cost
    else:
        return 0.4  # Low confidence - minimal cost

def estimate_detailed_cost(damages: List[dict]) -> dict:
    """
    Calculate detailed cost breakdown for damages
    """
    total_cost = 0
    cost_breakdown = []
    
    for damage in damages:
        damage_type = damage["type"].lower()
        confidence = damage["confidence"]
        
        # Get base cost for this damage type
        base_cost = DAMAGE_COST_MAP.get(damage_type, 300)
        
        # Calculate severity multiplier based on confidence
        severity_multiplier = get_damage_severity_multiplier(confidence)
        
        # Apply bounding box area multiplier (larger damage = more expensive)
        bbox = damage["bbox"]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_multiplier = min(2.0, 1.0 + (bbox_area / 100000))  # Cap at 2x
        
        # Calculate final cost
        final_cost = base_cost * severity_multiplier * area_multiplier
        final_cost = round(final_cost, 2)
        total_cost += final_cost
        
        cost_breakdown.append({
            "type": damage["type"],
            "confidence": confidence,
            "base_cost": base_cost,
            "severity_multiplier": round(severity_multiplier, 2),
            "area_multiplier": round(area_multiplier, 2),
            "final_cost": final_cost,
            "bbox_area": round(bbox_area, 2)
        })
    
    return {
        "total_cost": round(total_cost, 2),
        "cost_breakdown": cost_breakdown
    }

# Part-specific damage costs
PART_DAMAGE_COSTS = {
    'bonnet': {'scratch': 300, 'dent': 600, 'crack': 400, 'break': 800},
    'fender': {'scratch': 250, 'dent': 500, 'crack': 350, 'break': 700},
    'door': {'scratch': 280, 'dent': 550, 'crack': 380, 'break': 750},
    'bumper': {'scratch': 200, 'dent': 450, 'crack': 300, 'break': 600}
}

# Damage severity multipliers based on confidence
def get_damage_severity_multiplier(confidence: float) -> float:
    """Convert confidence to severity multiplier"""
    if confidence > 0.8:
        return 1.0  # High confidence - full cost
    elif confidence > 0.5:
        return 0.7  # Medium confidence - reduced cost
    else:
        return 0.4  # Low confidence - minimal cost

def estimate_detailed_cost(damages: List[dict]) -> dict:
    """
    Calculate detailed cost breakdown for damages
    """
    total_cost = 0
    cost_breakdown = []
    
    for damage in damages:
        damage_type = damage["type"].lower()
        confidence = damage["confidence"]
        
        # Calculate cost based on damage type and confidence
        if damage_type in PART_DAMAGE_COSTS:
            # If it's a vehicle part, use average part repair cost
            base_cost = DAMAGE_COST_MAP.get(damage_type, 300)
        else:
            # Try to match with damage types
            base_cost = 250  # default
        
        # Adjust cost based on confidence (higher confidence = more certain = higher cost)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
        adjusted_cost = base_cost * confidence_multiplier
        
        # Apply additional multipliers based on bounding box size (larger = more severe)
        bbox = damage["bbox"]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_multiplier = min(2.0, 1.0 + (bbox_area / 100000))  # Cap at 2x
        final_cost = adjusted_cost * area_multiplier
        
        # Round to reasonable amount
        final_cost = round(final_cost, 2)
        total_cost += final_cost
        
        cost_breakdown.append({
            "type": damage["type"],
            "confidence": confidence,
            "base_cost": base_cost,
            "confidence_multiplier": round(confidence_multiplier, 2),
            "area_multiplier": round(area_multiplier, 2),
            "final_cost": final_cost,
            "bbox_area": round(bbox_area, 2)
        })
    
    return {
        "total_cost": round(total_cost, 2),
        "cost_breakdown": cost_breakdown
    }
@app.get("/")
async def root():
    return {
        "message": "Vehicle Damage Assessment API",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "assess_damage": "/api/assess-damage (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/assess-damage")
async def assess_damage(
    pickup_images: List[UploadFile] = File(...),
    return_images: List[UploadFile] = File(...)
):
    """
    Compare pickup and return images to detect new damages
    """
    try:
        if not pickup_images or not return_images:
            raise HTTPException(status_code=400, detail="Both pickup and return images are required")
        
        print(f"🔄 Processing {len(pickup_images)} pickup images and {len(return_images)} return images")
        
        # Process pickup images
        pickup_damages = []
        for img in pickup_images:
            image_data = await img.read()
            damages = process_single_image(image_data, f"pickup_{img.filename}")
            pickup_damages.extend(damages)
        
        # Process return images
        return_damages = []
        for img in return_images:
            image_data = await img.read()
            damages = process_single_image(image_data, f"return_{img.filename}")
            return_damages.extend(damages)
        
        # Compare damages and find new ones
        comparison_result = compare_damages(pickup_damages, return_damages)
        
        # Generate report
        report = generate_damage_report(comparison_result)
        
        return JSONResponse(report)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    
def consolidate_damage_types(damages: List[dict]) -> List[dict]:
    """
    Consolidate similar damage types and choose the most confident one
    """
    if not damages:
        return []
    
    # Group damages by similar bounding boxes
    consolidated = []
    used_indices = set()
    
    for i, damage in enumerate(damages):
        if i in used_indices:
            continue
            
        similar_damages = [damage]
        
        # Find similar damages (overlapping bounding boxes)
        for j, other_damage in enumerate(damages[i+1:], i+1):
            if j in used_indices:
                continue
                
            if calculate_bbox_overlap(damage['bbox'], other_damage['bbox']) > 0.3:
                similar_damages.append(other_damage)
                used_indices.add(j)
        
        # Choose the damage with highest confidence from similar ones
        best_damage = max(similar_damages, key=lambda x: x['confidence'])
        consolidated.append(best_damage)
        used_indices.add(i)
    
    print(f"🔧 Type consolidation: {len(damages)} -> {len(consolidated)} damages")
    return consolidated 

def process_single_image(image_data: bytes, filename: str) -> List[dict]:
    """
    Process a single image and return detected damages with NMS applied
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return []
        
        # Run YOLO model
        results = model(image)
        
        damages = []
        for result in results:
            for box in result.boxes:
                damage_info = {
                    "type": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "image_source": filename
                }
                damages.append(damage_info)
        
        # Apply Non-Maximum Suppression to remove duplicates
        damages = apply_non_maximum_suppression(damages, iou_threshold=0.5)
        
        print(f"✅ Processed {filename}: Found {len(damages)} unique damages")
        return damages
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return []

def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate overlap between two bounding boxes (IoU - Intersection over Union)
    """
    try:
        # Fixed variable names - removed underscores that were causing the error
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    except Exception as e:
        print(f"Error calculating bbox overlap: {e}")
        return 0.0

def generate_damage_report(comparison_result: dict) -> dict:
    """
    Generate comprehensive damage report with detailed cost estimation
    """
    new_damages = comparison_result["new_damages"]
    
    # Calculate detailed costs
    cost_estimation = estimate_detailed_cost(new_damages)
    total_cost = cost_estimation["total_cost"]
    
    # Calculate severity score (0-10 scale)
    severity_score = 0
    for damage in new_damages:
        # Severity based on confidence and potential cost impact
        damage_severity = damage["confidence"] * 5  # 0-5 per damage
        severity_score += damage_severity
    
    # Normalize severity score
    severity_score = min(10, severity_score)
    
    # Damage summary by type
    damage_summary = {}
    for damage in new_damages:
        damage_type = damage["type"]
        damage_summary[damage_type] = damage_summary.get(damage_type, 0) + 1
    
    return {
        "assessment_id": f"assess_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "summary": {
            "new_damages_count": len(new_damages),
            "total_repair_cost": total_cost,
            "severity_score": round(severity_score, 1),
            "assessment_time": datetime.now().isoformat()
        },
        "damage_breakdown": damage_summary,
        "detailed_damages": new_damages,
        "cost_breakdown": cost_estimation["cost_breakdown"],  # Add detailed cost info
        "images_processed": {
            "pickup_images": comparison_result["total_pickup_damages"],
            "return_images": comparison_result["total_return_damages"]
        }
    }
    
def apply_non_maximum_suppression(damages: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """
    Remove duplicate detections using Non-Maximum Suppression (NMS)
    This ensures the same damage isn't detected multiple times with different labels
    """
    if not damages:
        return []
    
    # Sort damages by confidence (highest first)
    damages_sorted = sorted(damages, key=lambda x: x['confidence'], reverse=True)
    
    filtered_damages = []
    
    while damages_sorted:
        # Take the damage with highest confidence
        best_damage = damages_sorted.pop(0)
        filtered_damages.append(best_damage)
        
        # Find and remove duplicates that overlap too much
        damages_sorted = [
            damage for damage in damages_sorted 
            if calculate_bbox_overlap(best_damage['bbox'], damage['bbox']) < iou_threshold
        ]
    
    print(f"🔍 NMS: {len(damages)} -> {len(filtered_damages)} damages after removing duplicates")
    return filtered_damages    
    
def compare_damages(pickup_damages: List[dict], return_damages: List[dict]) -> dict:
    """
    Compare damages between pickup and return to find new damages
    """
    new_damages = []
    
    for return_damage in return_damages:
        is_new = True
        
        # Check if this damage exists in pickup images
        for pickup_damage in pickup_damages:
            # Check if same type and in similar location
            if (return_damage["type"] == pickup_damage["type"] and 
                calculate_bbox_overlap(return_damage["bbox"], pickup_damage["bbox"]) > 0.3):
                is_new = False
                break
        
        if is_new:
            new_damages.append(return_damage)
    
    print(f"🔍 Comparison: {len(pickup_damages)} pickup damages vs {len(return_damages)} return damages -> {len(new_damages)} new damages")
    return {
        "new_damages": new_damages,
        "total_pickup_damages": len(pickup_damages),
        "total_return_damages": len(return_damages)
    }
# Additional endpoint for single image analysis (bonus)
@app.post("/api/analyze-single")
async def analyze_single_image(image: UploadFile = File(...)):
    """
    Analyze a single image for damages
    """
    try:
        image_data = await image.read()
        damages = process_single_image(image_data, image.filename)
        
        return {
            "filename": image.filename,
            "damages_found": len(damages),
            "damages": damages
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add these imports at the top of main.py
import base64
from io import BytesIO

# Add these functions after your existing functions in main.py

def draw_bounding_boxes(image_data: bytes, damages: List[dict]) -> str:
    """
    Draw bounding boxes on image and return base64 encoded image
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("❌ Failed to decode image")
            return None
        
        # Convert BGR to RGB for better color display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes for each damage
        for damage in damages:
            bbox = damage["bbox"]
            damage_type = damage["type"]
            confidence = damage["confidence"]
            
            # Validate bbox coordinates
            if len(bbox) != 4:
                print(f"❌ Invalid bbox format: {bbox}")
                continue
                
            # Convert bbox coordinates to integers
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except (ValueError, TypeError) as e:
                print(f"❌ Error converting bbox to integers: {bbox}, error: {e}")
                continue
            
            # Validate coordinates are within image bounds
            height, width = image_rgb.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Skip if bbox is invalid
            if x1 >= x2 or y1 >= y2:
                print(f"❌ Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2})")
                continue
            
            # Choose color based on damage type
            colors = {
                'scratch': (255, 0, 0),      # Red
                'dent': (0, 0, 255),        # Blue
                'crack': (0, 255, 0),       # Green
                'broken_light': (255, 255, 0), # Cyan
                'damage': (255, 0, 255)     # Magenta
            }
            color = colors.get(damage_type.lower(), (255, 165, 0))  # Orange as default
            
            # Draw rectangle
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            label = f"{damage_type} ({confidence:.2f})"
            
            # Get text size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Ensure text doesn't go above image
            text_y = max(y1 - 5, label_size[1] + 10)
            
            # Draw background for text
            cv2.rectangle(image_rgb, 
                         (x1, text_y - label_size[1] - 10), 
                         (x1 + label_size[0], text_y), 
                         color, -1)
            
            # Draw text
            cv2.putText(image_rgb, label, (x1, text_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert back to bytes
        success, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        
        if not success:
            print("❌ Failed to encode image to JPEG")
            return None
            
        image_bytes = buffer.tobytes()
        
        # Convert to base64 for easy transmission
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return f"data:image/jpeg;base64,{encoded_image}"
    
    except Exception as e:
        print(f"❌ Error drawing bounding boxes: {e}")
        import traceback
        traceback.print_exc()
        return None
@app.post("/api/analyze-image-with-boxes")
async def analyze_image_with_boxes(image: UploadFile = File(...)):
    """
    Analyze a single image and return image with bounding boxes
    """
    try:
        image_data = await image.read()
        
        # Process image to get damages
        damages = process_single_image(image_data, image.filename)
        
        # Draw bounding boxes on image
        image_with_boxes = draw_bounding_boxes(image_data, damages)
        
        return {
            "filename": image.filename,
            "damages_found": len(damages),
            "damages": damages,
            "image_with_boxes": image_with_boxes
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare-with-visualization")
async def compare_with_visualization(
    pickup_images: List[UploadFile] = File(...),
    return_images: List[UploadFile] = File(...)
):
    """
    Compare images and return visualizations with bounding boxes
    """
    try:
        if not pickup_images or not return_images:
            raise HTTPException(status_code=400, detail="Both pickup and return images are required")
        
        print(f"🔄 Processing {len(pickup_images)} pickup and {len(return_images)} return images")
        
        # Process pickup images
        pickup_results = []
        for img in pickup_images:
            try:
                image_data = await img.read()
                damages = process_single_image(image_data, img.filename)
                image_with_boxes = draw_bounding_boxes(image_data, damages)
                
                pickup_results.append({
                    "filename": img.filename,
                    "damages": damages,
                    "image_with_boxes": image_with_boxes
                })
            except Exception as e:
                print(f"❌ Error processing pickup image {img.filename}: {e}")
                continue
        
        # Process return images
        return_results = []
        for img in return_images:
            try:
                image_data = await img.read()
                damages = process_single_image(image_data, img.filename)
                image_with_boxes = draw_bounding_boxes(image_data, damages)
                
                return_results.append({
                    "filename": img.filename,
                    "damages": damages,
                    "image_with_boxes": image_with_boxes
                })
            except Exception as e:
                print(f"❌ Error processing return image {img.filename}: {e}")
                continue
        
        # Compare damages to find new ones
        all_pickup_damages = []
        for result in pickup_results:
            all_pickup_damages.extend(result["damages"])
            
        all_return_damages = []
        for result in return_results:
            all_return_damages.extend(result["damages"])
        
        comparison_result = compare_damages(all_pickup_damages, all_return_damages)
        report = generate_damage_report(comparison_result)
        
        # Add visualization data to report
        report["visualizations"] = {
            "pickup_images": pickup_results,
            "return_images": return_results
        }
        
        print(f"✅ Assessment complete: {len(comparison_result['new_damages'])} new damages found")
        return JSONResponse(report)
    
    except Exception as e:
        print(f"❌ Error in compare-with-visualization: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
# Add these imports at the top of main.py
import base64
from io import BytesIO

# Add these functions after your existing functions in main.py

def draw_bounding_boxes(image_data: bytes, damages: List[dict]) -> str:
    """
    Draw bounding boxes on image and return base64 encoded image
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB for better color display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes for each damage
        for damage in damages:
            bbox = damage["bbox"]
            damage_type = damage["type"]
            confidence = damage["confidence"]
            
            # Convert bbox coordinates to integers
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on damage type
            colors = {
                'scratch': (255, 0, 0),      # Red
                'dent': (0, 0, 255),        # Blue
                'crack': (0, 255, 0),       # Green
                'broken_light': (255, 255, 0), # Cyan
                'damage': (255, 0, 255)     # Magenta
            }
            color = colors.get(damage_type.lower(), (255, 165, 0))  # Orange as default
            
            # Draw rectangle
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            label = f"{damage_type} ({confidence:.2f})"
            
            # Get text size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw background for text
            cv2.rectangle(image_rgb, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(image_rgb, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        image_bytes = buffer.tobytes()
        
        # Convert to base64 for easy transmission
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return f"data:image/jpeg;base64,{encoded_image}"
    
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
        return None

@app.post("/api/analyze-image-with-boxes")
async def analyze_image_with_boxes(image: UploadFile = File(...)):
    """
    Analyze a single image and return image with bounding boxes
    """
    try:
        image_data = await image.read()
        
        # Process image to get damages
        damages = process_single_image(image_data, image.filename)
        
        # Draw bounding boxes on image
        image_with_boxes = draw_bounding_boxes(image_data, damages)
        
        return {
            "filename": image.filename,
            "damages_found": len(damages),
            "damages": damages,
            "image_with_boxes": image_with_boxes
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare-with-visualization")
async def compare_with_visualization(
    pickup_images: List[UploadFile] = File(...),
    return_images: List[UploadFile] = File(...)
):
    """
    Compare images and return visualizations with bounding boxes
    """
    try:
        if not pickup_images or not return_images:
            raise HTTPException(status_code=400, detail="Both pickup and return images are required")
        
        # Process pickup images
        pickup_results = []
        for img in pickup_images:
            image_data = await img.read()
            damages = process_single_image(image_data, img.filename)
            image_with_boxes = draw_bounding_boxes(image_data, damages)
            
            pickup_results.append({
                "filename": img.filename,
                "damages": damages,
                "image_with_boxes": image_with_boxes
            })
        
        # Process return images
        return_results = []
        for img in return_images:
            image_data = await img.read()
            damages = process_single_image(image_data, img.filename)
            image_with_boxes = draw_bounding_boxes(image_data, damages)
            
            return_results.append({
                "filename": img.filename,
                "damages": damages,
                "image_with_boxes": image_with_boxes
            })
        
        # Compare damages to find new ones
        all_pickup_damages = []
        for result in pickup_results:
            all_pickup_damages.extend(result["damages"])
            
        all_return_damages = []
        for result in return_results:
            all_return_damages.extend(result["damages"])
        
        comparison_result = compare_damages(all_pickup_damages, all_return_damages)
        report = generate_damage_report(comparison_result)
        
        # Add visualization data to report
        report["visualizations"] = {
            "pickup_images": pickup_results,
            "return_images": return_results
        }
        
        return JSONResponse(report)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")    

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Important for deployment
        port=8000,
        reload=True  # Auto-reload during development
    )