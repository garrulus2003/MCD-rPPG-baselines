from argparser import parse_args
import frame_or_video.train
import ppg_models.train

def run(args=None):
    args_parsed = parse_args(args)
    
    if args_parsed.model_class in ['by_frame', 'by_video']:
        frame_or_video.train.run(args)
       
    if args_parsed.model_class == 'ppg':
        ppg_models.train.run(args)

if __name__ == '__main__':
    run()