IMAGO bundled Haar Cascade models
=================================

This folder bundles the Haar Cascade XML model files distributed with the
Python OpenCV package (cv2/data). They are included so the matching LAB 10
classical detection tools can load their models locally without downloading
model files at runtime.

Models include frontal/profile face, eyes, smile, full/upper/lower body,
cat face, and license-plate cascades, plus alternate face/eye cascades.

These are traditional Haar Cascade classifiers, not neural-network models,
and their detection quality varies with image conditions. They do not provide
semantic segmentation of individual body parts.

Upstream project: https://github.com/opencv/opencv
OpenCV license: Apache License 2.0. See the OpenCV repository's LICENSE file
for the license text and terms.
