# Inception Score for MNIST

Train a "perfect" MNIST classifier, and use it to computing inception score (ICP)

With our ICP implementation (pre-trained model saved in directory 'model'), the testing set of MNIST yields an score 
<img src="https://latex.codecogs.com/gif.latex?$\bf{9.8793~\pm~0.0614}$" />

Note that different pre-trained model may lead to slightly different inception score.

-----
### The Format of Generated Images 

The generated images are saved in a **mat** file, with a tensor named 'images' of **size [10000,784]**, where 10000 is the number of images, and 784 is the dimension of a flattened MNIST image.


If you have multiple checkout points (each is a mat file) saved in a folder, you may specify the directory as

```Python
    # folders for generated images
    result_folder = './example_dir/'

    icp = []
    for k in range(50):
        k = k + 1
        mat = scipy.io.loadmat(result_folder+ '{}.mat'.format(str(k).zfill(3)))
```

If you have one checkout point saved in a mat file, you may specify the file as

```Python
    file_name = 'example.mat'
    mat = scipy.io.loadmat(result_folder+ file_name )
```

-----

### How to Use the Code: Evaluation, Re-train and Plot

To evaluate the ICP of generated images, run:

    mnist_cnn_icp_eval.py
    
If you would like to re-train your classifier model, run:

    mnist_cnn_train_slim.py
    
    
If you would like to plot your inception scores for multiple checkout points, run:

    mnist_icp_plot.py
    
