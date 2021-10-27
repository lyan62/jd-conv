## Classification model

- Train intent classificaiton model:
    `python classify.py --train --config config.yml --device 6`

    - Crosswoz preprocess intent dataset: `data/crosswoz_data_splitted`


- Train pairwise classificaiton model:
  `python classify_pairs.py --train --config pair_config.yml --device 6`
  
    - Crosswoz pairwise dataset:
    `data/crosswoz_chat_pairs`
    



