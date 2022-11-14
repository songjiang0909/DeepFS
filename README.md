# DeepFS


This is the implementation of our paper "[Bridging Self-Attention and Time Series Decomposition for
Periodic Forecasting](https://songjiang0909.github.io/pdf/cikm_ts.pdf)", published at CIKM'22.



Data
-----------------

The original data are from [this repo](https://github.com/zhouhaoyi/Informer2020), credits and copyrights belong to the authors!


How to run?
-----------------


* Step1 (run):
	* `cd ./src`
	* `python main.py --arguement arguement_values`
	* See explanations for other arguements and parameters in `main.py`.

The prediction and trained models are stored under the `result` folder.



Contact
----------------------
Song Jiang <songjiang@cs.ucla.edu>


Bibtex
----------------------

```bibtex
@inproceedings{deepfs2022,
  title={Bridging Self-Attention and Time Series Decomposition for Periodic Forecasting},
  author={Song Jiang, Tahin Syed, Xuan Zhu, Joshua Levy, Boris Aronchik, Yizhou Sun},
  booktitle={Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
  year={2022}
}
```
