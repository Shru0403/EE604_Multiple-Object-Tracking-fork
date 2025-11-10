##  How to Run and Evaluate

1. **Download the MOT16 dataset**  
   - Get it from [MOT16 Official Page](https://motchallenge.net/data/MOT16/)  
   - Place it inside the `datasets/` folder as follows:
     ```
     datasets/
     └── MOT16/
         └── train/
             ├── MOT16-02/
             ├── MOT16-04/
             ├── MOT16-05/
             ├── MOT16-09/
             ├── MOT16-10/
             ├── MOT16-11/
             └── MOT16-13/
     ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. Run DeepSORT on all training sequences
   python scripts/run_deepsort_all.py

4. Evaluate results on all sequences
   python scripts/evaluate_mot16_train.py


