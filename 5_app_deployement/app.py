import argparse
import cv2
import numpy as np

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Handle an input stream")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args


def capture_stream(args):
    ### TODO: Handle image, video or webcam
    if args.i == "CAM":
        args.i = 0
    capture  = cv2.VideoCapture(args.i)
    capture.open(args.i)
    is_image = False
    if args.i[-4:-1]+args.i[-1] in [".tif",".bmp",".jpg",".jpeg",".png"]:
        is_image = True
    if is_image == True:
        while capture.isOpened():
            bol,frame = capture.read()
            resized_frame = cv2.resize(frame,(100,100))
            final = cv2.Canny(resized_frame, 100, 200)
            output_file_name = "output" + args.i[-4:-1]+args.i[-1]
            cv2.imwrite(output_file_name,final)
    video_writer = ('output.mp4', 0x00000021, 30, (100,100))
    if is_image == False:
        while capture.isOpened():
            bol,frame = capture.read()
            if bol == True:
                resized_frame = cv2.resize(frame,(100,100))
                final = cv2.Canny(resized_frame, 100, 200)
                video_writer.write(final)
            else:
                break
    video_writer.release()
    capture.release()
    cv2.destroyAllWindows()
    
    
    ### TODO: Get and open video capture
    
    ### TODO: Re-size the frame to 100x100
    
    ### TODO: Add Canny Edge Detection to the frame, 
    ###       with min & max values of 100 and 200
    ###       Make sure to use np.dstack after to make a 3-channel image

    ### TODO: Write out the frame, depending on image or video

    ### TODO: Close the stream and any windows at the end of the application


def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()
