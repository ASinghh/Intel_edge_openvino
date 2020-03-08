import argparse
### TODO: Load the necessary libraries
from openvino.inference_engine import IECore, IENetwork

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "/home/workspace/models"
    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    ### TODO: Load the Inference Engine API
    core_element = IECore()
    ### TODO: Load IR files into their related class
    file_name = model_xml.split(".")[0]
    param_file = file_name + ".bin"
    network_element = IENetwork(model=model_xml, weights=param_file)

    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    core_element.add_extension(CPU_EXTENSION, "CPU")

    ### TODO: Get the supported layers of the network
    supported_layers = core_element.query_network(network=network_element, device_name="CPU")
    all_layers = [layers for layers in network_element.layers.keys()]
    
    unsupported_layers = []
    
    for i in all_layers:
        if i not in supported_layers:
            unsupported_layers.append(i)
            

    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.
    if len(unsupported_layers) >0:
        print(unsupported_layers)
        return
    else:
        core_element.load_network(network_element, "CPU")
        
    ### TODO: Load the network into the Inference Engine

    print("IR successfully loaded into Inference Engine.")
    

    return


def main():
    args = get_args()
    load_to_IE(args.m)


if __name__ == "__main__":
    main()
