
**Stack** (Current tasks)

**Ideas**
- [ ] Train on all available data for the submission model (dont keep a val set)
- [ ] Error analysis of the images + products that my model is failing on
- [ ] Multimodal LLM feedback on generated result? + adjustments
- [ ] Explore external datasets

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
19 19:14 - (agent) Created A100 40GB VM on GCP (us-central1-a, europe was stocked out), verified GPU and SSH access
19 19:32 - Built evaluate.py with HTML report (mAP, overlays, error categories, heatmaps, confidence dist). Baseline: det_mAP=0.0096, final=0.0067
19 19:42 - Analyzed GT annotations: not all shelf products are annotated (e.g. Natreen missing from categories). Discussed mAP scoring mechanics — low-conf spam is nearly free, high-conf FPs are devastating, IoU@0.5 is lenient
19 19:46 - (agent) Uploaded data + train.py to A100 VM, installed deps, started YOLOv8x fine-tuning (100 epochs, imgsz=1280, batch=4). Training running ~4.5 it/s
19 19:53 - Error analysis: 3.6% recall, 77% FPs, model predicts COCO classes (bottle/book/bowl). Fine-tuning is the critical next step
19 19:59 - Decided on Approach 1: end-to-end YOLOv8x nc=357. Stable/simple first, split pipeline only if classification becomes bottleneck
19 19:55 - (agent) Moved training work to separate worktree (nm-1-training, branch: training). Training at epoch 34/100, mAP@0.5=0.677
19 19:59 - (agent) Training complete — early stopped at epoch 57, best at epoch 42. mAP@0.5=0.685, mAP@0.5:0.95=0.448 (up from 0.0096 baseline)
19 20:04 - (agent) Downloaded best.pt, created build_submission.sh (12 automated validation checks), built submission.zip (120MB, all checks pass)
19 21:59 - (agent) Re-read competition docs — found critical issue: trained with ultralytics 8.4.24 but sandbox has 8.1.0 (weights incompatible). Need to retrain pinned or export ONNX
19 22:19 - (agent) Fixed: retrained with ultralytics==8.1.0, removed torch.load patch from run.py, added security scan to build script. Training v2 running on A100
19 22:51 - (agent) Training v2 complete — mAP@0.5=0.682, mAP@0.5:0.95=0.453. Downloaded best.pt, rebuilt submission.zip (120MB, all checks pass). Ready to upload
19 23:22 - (agent) Ran 11 inference experiments: conf=0.01+TTA is best (det_mAP 0.883→0.902). Updated run.py. Launched 5 training experiments on A100 (YOLOv8l, imgsz=1536, heavy aug, lower LR, SGD)
20 01:17 - (agent) Phase 1 complete. Best: YOLOv8x trained at imgsz=1536 + conf=0.01 + NMS=0.5 → det_mAP=0.906 (+2.5% over baseline). Heavy aug/lower LR/SGD/YOLOv8l all worse. Submission rebuilt
20 09:41 - Checked gcloud quotas: 1 A100 active in us-central1, 200GB SSD used. GPU quotas generous (16 A100 on-demand, 64 spot per region). A100 80GB quota is 0 everywhere
20 09:41 - (agent) Switched to ONNX inference after .pt submission failed (torch.load weights_only issue). ONNX det_mAP=0.911 (better than .pt 0.906). No blocked imports, 212MB zip. Ready to submit
20 10:05 - Researched GCP AI services for competition: Vertex AI AutoML, Model Garden, Vision API, TPUs. Assessed feasibility against competition constraints (offline sandbox, 300s limit, L4 GPU)
20 10:14 - (agent) Mapped 326 product reference images to 321 categories. Generated 2724 synthetic training images (pasting cutouts onto shelf crops). Training YOLOv8x on augmented dataset (2885 images, imgsz=1536)
20 12:19 - (agent) Restructured repo: experiments/ per model, scripts/ shared. Created a100-vm-2 (us-central1-f). RT-DETR training launched (needed ultralytics 8.4.24, will ONNX export). Ref_aug at ep30 looks bad (mAP 0.44 vs 0.68 baseline) — synthetic data too artificial
20 13:31 - (agent) RT-DETR: total failure (mAP=0.000, 0 predictions) — transformer needs way more data than 248 images. Ref_aug at ep47: mAP=0.525, still bad. Both dead ends. Best remains yolo_imgsz1536 (det_mAP=0.911 via ONNX)
20 14:17 - Deep research on 10 improvement approaches: SAHI/tiling, WBF ensemble, two-stage detect+classify with DINOv2 embeddings, training tricks, template matching, few-shot, pseudo-labeling, cross-validation. Ranked by expected impact and feasibility
20 14:50 - (agent) SAHI tiling tested: det_mAP dropped 0.911→0.880 (too many FPs from tile boundaries). Reverted. Launched yolo_tuned (VM1) and yolo_fulldata (VM2). Next: WBF multi-scale ensemble
20 14:58 - (agent) WBF ensemble (1536+1280): det slightly worse (0.908 vs 0.911) but cls +1.6% (0.841 vs 0.825). Final score improved 0.885→0.888. Needs dynamic ONNX export
20 15:41 - (agent) yolo_tuned: worse (mAP 0.653 vs 0.682 baseline), killed. yolo_fulldata: done, ONNX exported. Built DINOv2 gallery (326 products, 454KB). Testing DINOv2 crop classification on val set
20 15:47 - (agent) DINOv2 classification: worse at all thresholds (cls dropped 0.825→0.807 even conservatively). Ref images too different from shelf crops. Dead end without fine-tuning
20 16:50 - (agent) Built submission with fulldata model + WBF ensemble (1536+1280). Verified on VM2 — 37K predictions, pipeline works. 212MB zip, all checks pass
20 18:45 - (agent) 3-scale WBF no better than 2-scale. Spun up VM3+VM4 (us-east1-b). Training 4 diverse fulldata models across 4 VMs for multi-model WBF ensemble (different seeds, LRs, imgsz)
20 20:01 - (agent) Submitted fulldata_v2 + WBF — competition score improved slightly. fulldata_v2 ONNX exported. VM1 ensemble_b near done, VM3/VM4 restarted after OpenCV fix. Preparing multi-model ensemble
20 20:09 - (agent) FP16 quantization: 131MB per model (half of FP32). Built 2-model WBF submission with model_1536 + model_1280 (fulldata_v2, FP16). 241MB zip, all checks pass. VM3/VM4 still training for potential 3rd model
20 23:34 - (agent) Multi-model WBF tested: 3-way ensemble (modelA@1536 + modelB@1536 + modelA@1280) = new best final=0.8902 (+0.5%). cls_mAP jumped to 0.8475. All VMs done. Need fulldata versions for submission
