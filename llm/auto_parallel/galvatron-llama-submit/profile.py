from paddlenlp.experimental.galvatron.profiler.model_profiler import ModelProfiler, ModelProfilerArguments
from paddlenlp.experimental.galvatron.utils import get_current_all_args

if __name__ == '__main__':
    args_dict = get_current_all_args()
    model_profiler_args = ModelProfilerArguments()
    model_profiler_args.initialize(args_dict=args_dict)
    model_profiler = ModelProfiler(model_profiler_args, args_dict)
    if model_profiler.args.profile_type == 'memory':
        model_profiler.launch_memory_profiling_static_scripts()
        model_profiler._process_single_sequence_static_config()
        exit(0)
    model_profiler.launch_profiling()
    model_profiler.process_data()