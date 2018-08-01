"""
Please refer the basic model to this link:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
Please refer batch normalized lstm to these links:
    https://arxiv.org/abs/1603.09025
    https://github.com/jihunchoi/recurrent-batch-normalization-pytorch.git
This script applies the basic LSTM algorithm in Pytorch
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_class import StockDataset
from model_class import LSTM, LSTMCell, BNLSTMCell

def main():
    DATA_PATH = args.data
    model_name = args.model
    save_dir = args.save
    HIDDEN_SIZE = args.hidden_size
    pmnist = args.pmnist
    BATCH_SIZE = args.batch_size
    max_iter = args.max_iter
    use_gpu = args.gpu
    
    torch.manual_seed(1)
     
    # Assign the path the the data pickle file
    device = torch.device("cuda:0" if use_gpu else "cpu")
    
    # List hyperparameters here
    TRAIN_SIZE = 1
    #TEST_SIZE = 2
    INPUT_DIM = 2
    #HIDDEN_DIM = 14
    PREDICT_DIM = 1
    #HISTORY_DIM = 180
    
#    if torch.cuda.is_available():
#        MINIBATCH_SIZE = int(BATCH_SIZE / torch.cuda.device_count())
#    else:
#        MINIBATCH_SIZE = BATCH_SIZE
    
    stock_dataset = StockDataset(DATA_PATH, BATCH_SIZE, device)
    stock_dataset.__getitem__(0)
    stock_dataloader = DataLoader(dataset=stock_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False)
    
    # For testing purposes, we only need to slice some number of samples out of
    # the Dataloader. We do not need to run through the whole dataset.
    data_iter = stock_dataloader.__iter__()
    
    
    ###########################################################################
    # Train the model:
    if model_name == 'bnlstm':
        model = LSTM(cell_class=BNLSTMCell, input_size=INPUT_DIM,
                     hidden_size=HIDDEN_SIZE, batch_first=True,
                     max_length=784)
    elif model_name == 'lstm':
        model = LSTM(cell_class=LSTMCell, input_size=INPUT_DIM,
                     hidden_size=HIDDEN_SIZE, batch_first=True)
    else:
        raise ValueError
    
#    
#    if use_gpu and torch.cuda.device_count() > 1:
#      print("Let's use", torch.cuda.device_count(), "GPUs!")
#      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#      model = nn.DataParallel(model, dim=0)
    
    model.to(device)
    
    # Use the mean square errorloss function to measures the distance of 
    # prediction from the actuall stock value
    hidden2out = nn.Linear(HIDDEN_SIZE, PREDICT_DIM)
    loss_function = nn.MSELoss(size_average=True, reduce=True)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # See what the scores are before training
    # =========================================================================
#    data_iter.__init__(stock_dataloader)
#    with torch.no_grad():
#        for item in range(TRAIN_SIZE):
#            hidden = model.module.init_hidden(BATCH_SIZE)
#            input_data, targets = data_iter.__next__()
#    #        input_data = input_data.transpose(0,1)
#            input_data.to(device)
#            targets.to(device)
#            tag_scores, hidden = model(input_data, hidden)
#            loss = loss_function(tag_scores, targets)
#            
#            print(loss)
    # =========================================================================
    start_time = time.time()
    for epoch in range(max_iter):
        print(epoch)
        data_iter.__init__(stock_dataloader)
        for item in range(TRAIN_SIZE): #test_tag:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            #hidden = model.init_hidden(BATCH_SIZE)
    
            # Step 2. Get our inputs ready for the network, that is, turn them 
            # into Tensors of word indices.
            input_data, targets, length = data_iter.__next__()
    #        input_data = input_data.transpose(0,1)
            
            # Step 3. Run our forward pass.
            hx = None
            if not pmnist:
                h0 = torch.normal(mean=0.0,
                                  std=torch.ones(input_data.size(0), 
                                                 HIDDEN_SIZE)/10)
                c0 = torch.normal(mean=0.0,
                                  std=torch.ones(input_data.size(0), 
                                                 HIDDEN_SIZE)/10)
                hx = (h0, c0)
            output, (_, _) = model(input_=input_data, length=length, hx=hx)
            prediction = hidden2out(output).squeeze()
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(prediction, targets)
            #print(loss)
            loss.backward()
            optimizer.step()
    
    print("--- %s seconds ---" % (time.time() - start_time))
    # See what the scores are after training
    # =========================================================================
#    data_iter.__init__(stock_dataloader)
#    with torch.no_grad():
#        for item in range(TRAIN_SIZE): #test_tag:
#            hidden = model.module.init_hidden(BATCH_SIZE)
#            input_data, targets = data_iter.__next__()
#    #        input_data = input_data.transpose(0,1)
#            input_data.to(device)
#            targets.to(device)
#            tag_scores, hidden = model(input_data, hidden)
#            loss = loss_function(tag_scores, targets)
#            
#            print(loss)
    # =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the model using MNIST dataset.')
    parser.add_argument('--data', default=
                        './data/Normalized_data/normalized_data_ver5.pickle', 
                        required=False,
                        help='The path to save MNIST dataset, or '
                             'the path the dataset is located')
    parser.add_argument('--model', default='lstm', required=False, 
                        choices=['lstm', 'bnlstm'],
                        help='The name of a model to use')
    parser.add_argument('--save', default='.', required=False,
                        help='The path to save model files')
    parser.add_argument('--hidden-size', default=3, required=False, 
                        type=int,
                        help='The number of hidden units')
    parser.add_argument('--pmnist', default=False, action='store_true',
                        help='If set, it uses permutated-MNIST dataset')
    parser.add_argument('--batch-size', default=8, required=False, type=int,
                        help='The size of each batch')
    parser.add_argument('--max-iter', default=100, required=False, type=int,
                        help='The maximum iteration count')
    parser.add_argument('--gpu', default=False, action='store_true',
                        help='The value specifying whether to use GPU')
    args = parser.parse_args()
    
    torch.set_default_tensor_type(torch.DoubleTensor)
    
    main()