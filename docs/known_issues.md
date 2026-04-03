
**Key improvement opportunities**:

1. **Detection ranking**: The detector sorts by area (largest first), but for `pidgeot` and `treecko` the correct card is a secondary detection. Ranking by contour quality (compactness, edge strength) rather than size would help.

2. **Photo-vs-scan normalization**: The biggest gap is comparing real photos (with lighting, shadows, texture) against small digital scans. Histogram equalization or CLAHE normalization before hashing would reduce this gap.

3. **Reference image quality**: Building the hash DB with `--small` images produces lower-quality reference hashes. Using large images would improve matching precision.

4. **Hash size tuning**: Current hash_size=16 (256-bit hashes) is very granular. Reducing to hash_size=8 (64-bit) would be more tolerant of photo-vs-scan differences, at the cost of less discrimination between similar cards.
