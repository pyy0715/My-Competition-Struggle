import fasttext
import os

if __name__ == '__main__':
   model = fasttext.train_supervised("./dataset/train.txt",
                                     lr=0.005,
                                     epoch=100,
                                     dim = 256,
                                     thread=4,
                                     ws = 1, 
                                     wordNgrams=1,
                                     minCount=3,
                                     )
   print(model.test("./dataset/valid.txt", k=-1))
   model.save_model("model_log.bin")
