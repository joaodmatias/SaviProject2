import random

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import nn
from torchvision import datasets, models, transforms


class ClassificationVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
        self.tensor_to_pil_image = transforms.ToPILImage()

    def draw(self, inputs, outputs, class_names):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(8,6)
        plt.suptitle(self.title)
        plt.legend(loc='best')

        inputs = inputs
        batch_size,_,_,_ = list(inputs.shape)

        
        output_probabilities = F.softmax(outputs, dim=1).tolist()
        
        print(output_probabilities)

        list_probabilities = []
        results = {}
        for i, output_probabilities_dog in enumerate(output_probabilities):
            
            output_probabilities_dog_max = max(output_probabilities_dog) 
            list_probabilities.append(output_probabilities_dog_max)
            max_index = np.argmax(output_probabilities_dog)
            print("lista posição: "+str(i)+" com o valor de: "+str(output_probabilities_dog_max)+" na posição interna de: "+str(max_index))
            results[i] = {"output_probabilities": output_probabilities_dog_max, "max_index": max_index}

        
        # for i in range(0,50):
        #     output_probabilities_dog = [x[i] for x in output_probabilities]
        #     for f in range(batch_size):
        #         if i == 0:
        #             output_probabilities_dog1 = output_probabilities_dog

        #         elif output_probabilities_dog[f] > output_probabilities_dog1[f]:
        #             output_probabilities_dog[f] = output_probabilities_dog1[f]
        #             print(output_probabilities_dog)
        #             e[f] = i
        #             print()

        for i in results:
            print("lista posição: "+str(i)+" com o valor de: "+str(results[i]["output_probabilities"])+" na posição interna de: "+str(results[i]["max_index"]))
        
        random_idxs = random.sample(list(range(batch_size)), k=1)
        # for plot_idx, image_idx in enumerate(random_idxs):
        for spot, x in enumerate(random_idxs):
            # for i in results:
            output_probability_dog = results[x]["output_probabilities"]
            max_index = results[x]["max_index"]
            #print("test: "+ str(max_index))

            
            image_t = inputs[x,:,:,:]
            image_pil = self.tensor_to_pil_image(image_t)

            ax = self.figure.add_subplot(5,5,spot+1) # define a 5 x 5 subplot matrix
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])


            title = class_names[max_index]
            title += ' ' + str(output_probability_dog)
            ax.set_xlabel(title, color='green')

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)


