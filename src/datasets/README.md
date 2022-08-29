# Dataset Processing 

TODO:

# Checking Duplicates 

### [PolyCoder](https://github.com/VHellendoorn/Code-LMs)

Dataset: https://zenodo.org/record/6363556

#### Overlap with Defects4j
- High amounts of overlap
- Reason is that training repos contains lots of duplicated defects4j code
  - For example: https://github.com/KTH/sequencer and https://github.com/SpoonLabs/coming are a part of the training dataset containing some multiple different patches for Defects4j
  - E.g., [Chart-1 Patch](https://github.com/SpoonLabs/coming/blob/master/src/main/resources/Defects4J_all_pairs/Chart_1/AbstractCategoryItemRenderer/Chart_1_AbstractCategoryItemRenderer_t.java)  
  - E.g., [Chart-1 Patch](https://github.com/KTH/sequencer/blob/master/results/Defects4J_patches/Chart_1/13_correct/AbstractCategoryItemRenderer.java)