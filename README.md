# Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch

### 7.30 Fixed bug ,added to_binary function

### 12.25 Polish

If you find any bugs, please contact me!!!

## Introduction

This repository is based on the paper [Symbolic Music Genre Transfer with CycleGAN](https://arxiv.org/pdf/1809.07575.pdf) and the repository  [CycleGAN-Music-Style-Transfer-Refactorization](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization)

## Installation

- Clone this repo:

  ```bash
  git clone https://github.com/Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch.git 
  cd Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch
  ```

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```



## Usage

### Prepare the dataset

Download the [dataset](https://drive.google.com/file/d/1zyN4IEM8LbDHIMSwoiwB6wRSgFyz7MEH/view) which can be use directly

Put them into the folder `traindata`, make sure they're like this:

```sh
-- Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch

	-- traindata

		|- JCP_mixed

		|- CP_P

		|- CP_C

		|- JP_P

		|- JP_J

		|- JC_C

		|- JC_J
```

For those who want to use their own datasets, or want to try all the preprocessing steps, please take a look at Testfile.py and convert_clean.py files which are in this [repo](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer). There are 6 steps in the Testfile.py which are remarked clearly. You can just comment/uncomment and run these code blocks in order. The second step is to run convert_clean.py. Please make sure all the directory paths have correct settings though it might take some time! It's very important! In addition, please ensure there are the same number of phrases (always downsampling) for each genre in a genre pair for the sake of avoiding imbalance. E.g., for classic vs. jazz, you might get 1000 phrases for classic and 500 phrases for jazz, then you should downsample the phrases for classic to 500.

### Train

use `sh train.sh` to simply train the model.

or 

```shell
python train.py --help
python train.py --epoch 5 --batch-size 2 --model-name CP --data-mode full
```

I found that the model in the later stages of training produced almost the same test results.I think the `Generator` learned the discriminant mode of `Discriminator`.So I suggest that try to increase the weight of `cycle loss` and  `mixed loss`( which makes the `Discriminator` learn more discriminant mode ).

You will get the `.pth` model file in the folder `saved_models/model_name`

### Test

```shell
python test.py --data-dir path/to/data --model-dir path/to/model --batch-size 2 --model-name CP --test-mode A2B
```

The model have the mode `A2B` and `B2A` for testing, which are supposed to Indicate the first music genre to second music genre. 

You will get the test results in folder `test`.

