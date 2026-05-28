import os
import json
import shutil
import logging
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from deepface import DeepFace
from django.conf import settings
from django.db import transaction
from app.models import Person, PersonImage, RecognitionLog

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = '.embedding_cache.json'

def get_cache_path():
    return os.path.join(settings.MEDIA_ROOT, 'unknown', CACHE_FILE_NAME)

def load_embedding_cache():
    cache_path = get_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading embedding cache: {e}")
    return {}

def save_embedding_cache(cache):
    cache_path = get_cache_path()
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving embedding cache: {e}")

def classify_unknown_faces():
    """
    Scans the unknown folder, extracts embeddings using DeepFace,
    groups them using DBSCAN, and returns a structured list of groups.
    """
    unknown_dir = os.path.join(settings.MEDIA_ROOT, 'unknown')
    os.makedirs(unknown_dir, exist_ok=True)
    
    # 1. Scan for valid images
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_files = os.listdir(unknown_dir)
    image_filenames = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    if not image_filenames:
        return []
        
    # 2. Load cached embeddings
    cache = load_embedding_cache()
    
    embeddings = {}
    no_face_images = []
    
    # Track cache hits and misses
    cache_updated = False
    
    # 3. Extract embeddings
    for filename in image_filenames:
        img_path = os.path.join(unknown_dir, filename)
        
        # Check cache first
        if filename in cache:
            # Cache stores None or "NO_FACE" for images without faces
            val = cache[filename]
            if val == "NO_FACE" or val is None:
                no_face_images.append(filename)
            else:
                embeddings[filename] = val
            continue
            
        # Cache miss: Explicit OpenCV -> Crop -> ArcFace Pipeline
        try:
            # 1. 📸 Raw Image -> 🎯 OpenCV Haar Cascade Detector
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError("Could not read image")
                
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Using OpenCV's built-in frontal face cascade (fastest for CPU)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces (tuned for CCTV: small faces, varied lighting)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            if len(faces) == 0:
                # No face found by OpenCV
                no_face_images.append(filename)
                cache[filename] = "NO_FACE"
                continue
                
            # 2. ✂️ Crop Face (take the largest face if multiple)
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Add a slight padding to the crop (10%)
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_bgr.shape[1], x + w + pad_x)
            y2 = min(img_bgr.shape[0], y + h + pad_y)
            
            cropped_face = img_bgr[y1:y2, x1:x2]
            
            # 3. 🧠 ArcFace -> 📊 512-D Embedding
            # We pass the pre-cropped numpy array and skip DeepFace's internal detector
            res = DeepFace.represent(img_path=cropped_face, model_name='ArcFace', enforce_detection=False, detector_backend='skip', align=False)
            
            if res and len(res) > 0 and 'embedding' in res[0]:
                emb = res[0]['embedding']
                if isinstance(emb, list) and len(emb) > 0:
                    embeddings[filename] = emb
                    cache[filename] = emb
                else:
                    no_face_images.append(filename)
                    cache[filename] = "NO_FACE"
            else:
                no_face_images.append(filename)
                cache[filename] = "NO_FACE"
                
        except Exception as e:
            logger.error(f"Pipeline extraction failed for {filename}: {e}")
            no_face_images.append(filename)
            cache[filename] = "NO_FACE"
        
        cache_updated = True
        
    # Save cache if we computed any new embeddings
    if cache_updated:
        save_embedding_cache(cache)
        
    # 4. Group embeddings using DBSCAN
    groups_list = []
    
    if embeddings:
        filenames_with_embeddings = list(embeddings.keys())
        embeddings_matrix = np.array([embeddings[f] for f in filenames_with_embeddings])
        
        # DBSCAN: cosine distance metric, eps=0.30, min_samples=2
        # ArcFace produces 512-d embeddings. Cosine distance thresholds:
        #   Same person: 0.0 - 0.30 | Different person: 0.40+
        # eps=0.30 ensures only genuinely same-person faces cluster together.
        if len(embeddings_matrix) >= 2:
            try:
                db = DBSCAN(eps=0.30, min_samples=2, metric='cosine')
                labels = db.fit_predict(embeddings_matrix)
            except Exception as e:
                logger.error(f"DBSCAN clustering failed: {e}")
                labels = np.array([-1] * len(embeddings_matrix))
        else:
            # Only 1 embedding, cannot run DBSCAN with min_samples=2
            labels = np.array([-1] * len(embeddings_matrix))
            
        # Group by labels
        clusters = {}
        outliers = []
        
        for idx, label in enumerate(labels):
            filename = filenames_with_embeddings[idx]
            if label != -1:
                label_id = int(label)
                if label_id not in clusters:
                    clusters[label_id] = []
                clusters[label_id].append(filename)
            else:
                outliers.append(filename)
                
        # Format clusters
        for cluster_id, files in clusters.items():
            groups_list.append({
                'id': f"group_{cluster_id}",
                'images': [{'filename': f, 'url': f"{settings.MEDIA_URL}unknown/{f}"} for f in files],
                'count': len(files),
                'is_cluster': True,
                'is_noface': False
            })
            
        # Format outliers as singletons (each gets its own single group so they can be named individually)
        for idx, f in enumerate(outliers):
            groups_list.append({
                'id': f"single_{idx}",
                'images': [{'filename': f, 'url': f"{settings.MEDIA_URL}unknown/{f}"}],
                'count': 1,
                'is_cluster': False,
                'is_noface': False
            })
            
    # 5. Add "No Face Detected" group if present
    if no_face_images:
        groups_list.append({
            'id': 'noface',
            'images': [{'filename': f, 'url': f"{settings.MEDIA_URL}unknown/{f}"} for f in no_face_images],
            'count': len(no_face_images),
            'is_cluster': False,
            'is_noface': True
        })
        
    return groups_list

def assign_group_to_person(group_images, person_name):
    """
    Associates the list of unknown images to a person:
    - Creates or retrieves the Person record in the DB
    - Moves physical files from media/unknown to media/dataset/<person_name>/
    - Creates PersonImage records in the DB
    - Deletes corresponding RecognitionLog entries
    - Cleans cache entries for moved images
    """
    if not group_images or not person_name:
        return False, "Invalid parameters."
        
    person_name = person_name.strip()
    if not person_name:
        return False, "Person name cannot be blank."
        
    unknown_dir = os.path.join(settings.MEDIA_ROOT, 'unknown')
    dataset_dir = os.path.join(settings.MEDIA_ROOT, 'dataset', person_name)
    
    # Load cache so we can clean up entries
    cache = load_embedding_cache()
    cache_updated = False
    
    try:
        with transaction.atomic():
            # 1. Create or retrieve the Person
            person, created = Person.objects.get_or_create(name=person_name)
            
            # 2. Ensure destination dataset folder exists
            os.makedirs(dataset_dir, exist_ok=True)
            
            for filename in group_images:
                src_path = os.path.join(unknown_dir, filename)
                
                # Check if file actually exists
                if not os.path.exists(src_path):
                    logger.warning(f"File {src_path} does not exist. Skipping.")
                    continue
                    
                # Create a unique target filename to prevent overwriting
                base_name, ext = os.path.splitext(filename)
                target_filename = filename
                target_path = os.path.join(dataset_dir, target_filename)
                
                counter = 1
                while os.path.exists(target_path):
                    target_filename = f"{base_name}_{counter}{ext}"
                    target_path = os.path.join(dataset_dir, target_filename)
                    counter += 1
                    
                # 3. Move the physical file
                shutil.move(src_path, target_path)
                
                # 4. Create the PersonImage record
                # Note: Django's ImageField needs relative path: dataset/<name>/<filename>
                relative_image_path = f"dataset/{person_name}/{target_filename}"
                PersonImage.objects.create(person=person, image=relative_image_path)
                
                # 5. Delete old RecognitionLog entries matching this image
                # The image_path in RecognitionLog is saved as 'unknown/<filename>'
                old_log_path = f"unknown/{filename}"
                RecognitionLog.objects.filter(image_path=old_log_path).delete()
                
                # 6. Delete cache entry
                if filename in cache:
                    del cache[filename]
                    cache_updated = True
                    
        # Save updated cache if any entries were removed
        if cache_updated:
            save_embedding_cache(cache)
            
        return True, f"Successfully assigned {len(group_images)} images to '{person_name}'."
        
    except Exception as e:
        logger.error(f"Error assigning group to person: {e}")
        return False, f"Failed to assign group: {str(e)}"
