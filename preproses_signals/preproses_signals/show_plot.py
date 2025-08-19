import matplotlib.pyplot as plt
def get_plot(list_sinales , list_labeles , size_plot = (120 * 3, 6), list_sacale = None, list_alpha=None, alphaes = 0.2 , Fs = 250):
    len_data = len(list_sinales[0])
    second = len_data / Fs
    plt.figure(figsize=size_plot)
    for i in range(len(list_sinales)):
        if list_alpha == None:
          if i  in [0,1]:
            alpha = alphaes
          else:
            alpha = 1
        else :
          alpha = list_alpha[i]


        max = np.max(list_sinales[i])
        min = np.min(list_sinales[i])
        list_sinales[i] = (list_sinales[i] - min) / (max - min)
        if list_sacale is not None:
          list_sinales[i] = list_sinales[i]  * list_sacale[i]
        time = np.linspace(0, second, len(list_sinales[i]))
        plt.plot(time , list_sinales[i], label=list_labeles[i], alpha = alpha)
    if len(list_labeles) == 3:
      plt.title(list_labeles[2])

    plt.xlabel('second')
    plt.ylabel('singnall Value')
    plt.legend()
    plt.show()





def show_plotes(dic_data,size_plot= (120 * .2 , 6),  list_sacale = None, list_alpha = None, alphaes = 0.2):
  for label, signall in dic_data:
    get_plot( [signall] , [label] , size_plot , list_sacale = None, list_alpha=None)