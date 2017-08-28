# Inception Score for MNIST

Training a MNIST classifier, and use it to computing inception score (ICP)

Under our ICP implementation, the testing set of MNIST yields an score 
<img src="https://latex.codecogs.com/gif.latex?$\mathbf{9.8793~\pm~0.0614}$" />

To evaluate the ICP of generated images, run:

    mnist_cnn_icp_eval.py
    
    
If you would like to re-train your classifier model, run:

    mnist_cnn_train_slim.py
    
    
<img src="icp_plot.pdf" data-canonical-src="icp_plot.pdf" width="460" height="250" />
