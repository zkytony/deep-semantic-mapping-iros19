from pprint import pprint
from deepsm.graphspn.tests.tbm.runner import get_noisification_level


if __name__ == "__main__":


    settings =  [[(0.5, 0.7), (0.0004, 0.00065), (0.5, 0.7), (0.17, 0.22)],
                 [(0.5, 0.7), (0.002, 0.008), (0.5, 0.7), (0.22, 0.40)],
                 [(0.5, 0.7), (0.014, 0.029), (0.5, 0.7), (0.22, 0.35)],
                 [(0.5, 0.7), (0.04, 0.09001), (0.5, 0.7), (0.22, 0.315)],
                 [(0.5, 0.7), (0.07, 0.13), (0.5, 0.7), (0.17, 0.22)],
                 [(0.5, 0.7), (0.17, 0.234), (0.5, 0.7), (0.13, 0.155)]]

    for i, setting in enumerate(settings):

        print("Noise Level %d" % i)
        nl = get_noisification_level(setting[0],
                                     setting[1],
                                     setting[2],
                                     setting[3],
                                     uniform_for_incorrect=i==0)

        pprint(nl)
        print("--------------")


          

          
                                                                                               

                                                                                               
          
                                                                                               
          
                                                                                               
          
