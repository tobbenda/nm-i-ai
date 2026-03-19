
**Stack** (Current tasks)

**Ideas**



___
## Log
19 18:29 - Fetched and reviewed all NorgesGruppen object detection challenge docs (overview, submission, scoring, examples)
19 18:30 - Problem understanding: detect+classify grocery products on shelves, scored 70% detection mAP + 30% classification mAP
19 18:31 - Clarified: 300s total for all images, ~88 products/image, L4 GPU forced in sandbox, 3 submissions/day
19 18:32 - Discussed scoring strategy: detection (70%) is priority, classification (30%) is bonus. Detection-only can score 0.70 max
19 18:33 - No starter code in repo yet, discussed setting up project structure
19 18:35 - Analyzed scoring strategy: detection (70%) is priority, fine-tune YOLOv8m/l/x with nc=357, watch bbox format and version pinning
19 18:37 - Created minimal architecture diagram: train.py → best.pt + run.py → zip → sandbox → score
19 18:40 - Extracted COCO dataset, reviewed example shelf images — dense products, similar appearances, varying angles
19 18:43 - Analyzed annotations: only front-facing visible products are boxed, ~99 per image, some overlap between adjacent items
19 18:46 - Reviewed product reference images — clean cutouts, 7 angles each, 345 products. Some products (o.b., Maretti) outside the 4 training sections
19 19:06 - Built and verified run.py with pretrained YOLOv8n — pipeline works, 19 detections on 3 test images, output format correct
19 19:10 - Upgraded to YOLOv8x + imgsz=1280, jumped from 19→48 detections on 3 images (~18% recall vs ~270 actual). No more quick fixes without fine-tuning
19 19:12 - Created submission.zip (131 MB) — verified against all submission requirements, ready to upload
19 19:14 - Sandbox simulation passed — unzipped, ran from clean dir, output format validated
19 19:14 - Created A100 40GB VM on GCP (us-central1-a, europe was stocked out), verified GPU and SSH access
