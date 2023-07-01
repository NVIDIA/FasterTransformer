mkdir hf_models

git clone https://huggingface.co/gpt2
cd gpt2
git lfs pull
cd ..
mv gpt2 hf_models

python hf_gpt2.py

```
tensor([[15496,    11,   616,  3290,   318, 13779]])
tensor([[15496,    11,   616,  3290,   318, 13779,    13,   314,  1101,   407,
          1654,   611,   673,   338,   257, 26188,   393,   407,    13,   314,
          1101,   407,  1654,   611,   673,   338,   257,  3290,   393,   407,
            13,   314,  1101,   407,  1654,   611,   673,   338]])
Hello, my dog is cute. I'm not sure if she's a puppy or not. I'm not sure if she's a dog or not. I'm not sure if she's
```

bash build_release.sh
cd build_release
./bin/multi_gpu_gpt_example

cat out

```
15496 11 616 3290 318 13779 13 314 1101 407 1654 611 673 338 257 26188 393 407 13 314 1101 407 1654 611 673 338 257 3290 393 407 13 314 1101 407 1654 611 673 338 
```



