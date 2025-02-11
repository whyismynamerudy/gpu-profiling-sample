import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import GPUtil
from datetime import datetime
import json
import os
import argparse
from prometheus_client import start_http_server, Gauge, Counter, Summary

class WorkloadProfiler:
    def __init__(self, output_dir="/app/profiling_results", prometheus_port=8000):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Prometheus metrics
        start_http_server(prometheus_port)
        self._setup_prometheus_metrics()
        
        # Traditional metrics storage
        self.metrics = {
            "timestamps": [],
            "gpu_utilization": [],
            "gpu_memory_used": [],
            "cpu_utilization": [],
            "system_memory_used": [],
            "inference_times": []
        }
    
    def _setup_prometheus_metrics(self):
        # Performance metrics
        self.inference_time = Summary('ml_inference_time_seconds', 
                                    'Time taken for inference')
        self.inference_count = Counter('ml_inference_total', 
                                     'Total number of inferences performed')
        
        # Resource utilization metrics
        self.gpu_utilization = Gauge('ml_gpu_utilization_percent', 
                                   'GPU utilization percentage')
        self.gpu_memory_used = Gauge('ml_gpu_memory_used_bytes', 
                                   'GPU memory used in bytes')
        self.cpu_utilization = Gauge('ml_cpu_utilization_percent', 
                                   'CPU utilization percentage')
        self.system_memory = Gauge('ml_system_memory_used_percent', 
                                 'System memory utilization percentage')
        
        # Model metrics
        self.model_load_time = Gauge('ml_model_load_time_seconds', 
                                   'Time taken to load model')
        self.gpu_power = Gauge('ml_gpu_power_watts', 
                             'GPU power consumption in watts')
        self.gpu_temperature = Gauge('ml_gpu_temperature_celsius', 
                                   'GPU temperature in Celsius')
        
        # Token metrics
        self.tokens_processed = Counter('ml_tokens_processed_total', 
                                      'Total number of tokens processed')
        self.token_rate = Gauge('ml_token_rate_per_second', 
                              'Token processing rate per second')

    def _collect_metrics(self):
        # Collect GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Using first GPU
            self.metrics["gpu_utilization"].append(gpu.load * 100)
            self.metrics["gpu_memory_used"].append(gpu.memoryUsed)
            
            # Update Prometheus GPU metrics
            self.gpu_utilization.set(gpu.load * 100)
            self.gpu_memory_used.set(gpu.memoryUsed * 1024 * 1024)  # Convert to bytes
            # Only set power and temperature if available
            if hasattr(gpu, 'temperature'):
                self.gpu_temperature.set(gpu.temperature)
            else:
                self.gpu_temperature.set(0)
            
            # Use nvidia-smi for power metrics since GPUtil doesn't provide them
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True)
                if result.returncode == 0:
                    power_draw = float(result.stdout.strip())
                    self.gpu_power.set(power_draw)
                else:
                    self.gpu_power.set(0)
            except:
                self.gpu_power.set(0)
        else:
            self.metrics["gpu_utilization"].append(0)
            self.metrics["gpu_memory_used"].append(0)
            self.gpu_utilization.set(0)
            self.gpu_memory_used.set(0)
            self.gpu_power.set(0)
            self.gpu_temperature.set(0)
        
        # Collect CPU and system metrics
        cpu_util = psutil.cpu_percent()
        mem_util = psutil.virtual_memory().percent
        
        self.metrics["cpu_utilization"].append(cpu_util)
        self.metrics["system_memory_used"].append(mem_util)
        
        # Update Prometheus CPU and memory metrics
        self.cpu_utilization.set(cpu_util)
        self.system_memory.set(mem_util)
        
        self.metrics["timestamps"].append(datetime.now().isoformat())

    def save_metrics(self):
        # Calculate summary statistics
        summary = {
            "avg_gpu_utilization": sum(self.metrics["gpu_utilization"]) / len(self.metrics["gpu_utilization"]),
            "max_gpu_utilization": max(self.metrics["gpu_utilization"]),
            "avg_gpu_memory": sum(self.metrics["gpu_memory_used"]) / len(self.metrics["gpu_memory_used"]),
            "max_gpu_memory": max(self.metrics["gpu_memory_used"]),
            "avg_inference_time": sum(self.metrics["inference_times"]) / len(self.metrics["inference_times"]),
            "total_inferences": len(self.metrics["inference_times"])
        }
        
        # Save both detailed metrics and summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/profiling_metrics_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                "summary": summary,
                "detailed_metrics": self.metrics
            }, f, indent=2)
        
        print(f"Metrics saved to {filename}")
        print("\nSummary Statistics:")
        for key, value in summary.items():
            print(f"{key}: {value:.2f}")

def setup_logging(output_dir):
    """Setup logging to both console and file"""
    import sys
    from datetime import datetime
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/ml_workload_{timestamp}.log"
    
    # Create a custom logger that writes to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    return log_file

def run_inference_workload(
    model_name="facebook/opt-350m",
    run_duration=60,  # Run duration in seconds
    input_text="Explain the theory of relativity in simple terms:",
    max_new_tokens=100,
    output_dir="/app/profiling_results",
    prometheus_port=8000
):
    # Setup logging
    log_file = setup_logging(output_dir)
    profiler = WorkloadProfiler(output_dir=output_dir, prometheus_port=prometheus_port)
    
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto"  # Automatically handle device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    load_time = time.time() - start_time
    profiler.model_load_time.set(load_time)  # Record model load time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_tokens = len(tokenizer.encode(input_text))
    
    # Run inferences for the specified duration
    end_time = start_time + run_duration
    inference_count = 0
    
    print(f"Running inferences for {run_duration} seconds...")
    while time.time() < end_time:
        profiler._collect_metrics()  # Collect metrics before inference
        
        inference_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        inference_time = time.time() - inference_start
        
        # Calculate tokens generated
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_tokens = len(tokenizer.encode(output_text))
        total_tokens = input_tokens + output_tokens
        
        # Update metrics
        profiler.inference_time.observe(inference_time)
        profiler.inference_count.inc()
        profiler.tokens_processed.inc(total_tokens)
        profiler.token_rate.set(total_tokens / inference_time)
        profiler.metrics["inference_times"].append(inference_time)
        
        inference_count += 1
        if inference_count % 10 == 0:  # Progress update every 10 inferences
            elapsed = time.time() - start_time
            remaining = max(0, run_duration - elapsed)
            print(f"Completed {inference_count} inferences. {remaining:.1f} seconds remaining")
    
    profiler.save_metrics()
    total_time = time.time() - start_time
    print(f"Workload completed! Ran {inference_count} inferences in {total_time:.1f} seconds")
    print(f"Average inference rate: {inference_count/total_time:.2f} inferences/second")
    print(f"Prometheus metrics available at http://localhost:{prometheus_port}/metrics")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ML inference workload')
    parser.add_argument('--output-dir', default="/app/profiling_results",
                      help='Directory to save profiling results')
    parser.add_argument('--model-name', default="facebook/opt-350m",
                      help='Model to use for inference')
    parser.add_argument('--run-duration', type=int, default=60,
                      help='Duration to run inferences in seconds')
    parser.add_argument('--input-text', 
                      default="Explain the theory of relativity in simple terms:",
                      help='Input text for inference')
    parser.add_argument('--max-new-tokens', type=int, default=100,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--prometheus-port', type=int, default=8000,
                      help='Port for Prometheus metrics')
    
    args = parser.parse_args()
    
    run_inference_workload(
        model_name=args.model_name,
        run_duration=args.run_duration,
        input_text=args.input_text,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        prometheus_port=args.prometheus_port
    )

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import time
# import psutil
# import GPUtil
# from datetime import datetime
# import json
# import os
# import argparse

# class WorkloadProfiler:
#     def __init__(self, output_dir):
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.metrics = {
#             "timestamps": [],
#             "gpu_utilization": [],
#             "gpu_memory_used": [],
#             "cpu_utilization": [],
#             "system_memory_used": [],
#             "inference_times": []
#         }
        
#     def _collect_metrics(self):
#         gpus = GPUtil.getGPUs()
#         if gpus:
#             gpu = gpus[0]  # Using first GPU
#             self.metrics["gpu_utilization"].append(gpu.load * 100)
#             self.metrics["gpu_memory_used"].append(gpu.memoryUsed)
#         else:
#             self.metrics["gpu_utilization"].append(0)
#             self.metrics["gpu_memory_used"].append(0)
            
#         self.metrics["cpu_utilization"].append(psutil.cpu_percent())
#         self.metrics["system_memory_used"].append(psutil.virtual_memory().percent)
#         self.metrics["timestamps"].append(datetime.now().isoformat())

#     def save_metrics(self):
#         # Save detailed metrics
#         metrics_file = os.path.join(self.output_dir, f"ml_metrics_{self.run_timestamp}.json")
#         with open(metrics_file, 'w') as f:
#             json.dump(self.metrics, f, indent=2)
#         print(f"Detailed metrics saved to {metrics_file}")
        
#         # Calculate and save summary statistics
#         summary = {
#             "avg_gpu_utilization": sum(self.metrics["gpu_utilization"]) / len(self.metrics["gpu_utilization"]),
#             "max_gpu_utilization": max(self.metrics["gpu_utilization"]),
#             "avg_gpu_memory": sum(self.metrics["gpu_memory_used"]) / len(self.metrics["gpu_memory_used"]),
#             "max_gpu_memory": max(self.metrics["gpu_memory_used"]),
#             "avg_inference_time": sum(self.metrics["inference_times"]) / len(self.metrics["inference_times"]),
#             "total_inferences": len(self.metrics["inference_times"]),
#             "timestamp": self.run_timestamp
#         }
        
#         summary_file = os.path.join(self.output_dir, f"ml_summary_{self.run_timestamp}.json")
#         with open(summary_file, 'w') as f:
#             json.dump(summary, f, indent=2)
#         print(f"Summary saved to {summary_file}")

# def run_inference_workload(
#     model_name="facebook/opt-350m",
#     num_inferences=50,
#     input_text="Explain the theory of relativity in simple terms:",
#     max_new_tokens=100,
#     output_dir="/app/profiling_results"
# ):
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     profiler = WorkloadProfiler(output_dir)
    
#     print(f"Loading model: {model_name}")
#     start_time = time.time()
    
#     # Load model and tokenizer
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
    
#     load_time = time.time() - start_time
#     print(f"Model loaded in {load_time:.2f} seconds")
    
#     # Prepare input
#     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
#     print(f"Running {num_inferences} inferences...")
#     for i in range(num_inferences):
#         profiler._collect_metrics()
        
#         inference_start = time.time()
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 num_return_sequences=1,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         inference_time = time.time() - inference_start
#         profiler.metrics["inference_times"].append(inference_time)
        
#         if i % 10 == 0:
#             print(f"Completed {i+1}/{num_inferences} inferences")
    
#     profiler.save_metrics()
#     print("Workload completed!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run ML inference workload')
#     parser.add_argument('--output-dir', required=True,
#                       help='Directory to save profiling results')
#     parser.add_argument('--model-name', default="facebook/opt-350m",
#                       help='Model to use for inference')
#     parser.add_argument('--num-inferences', type=int, default=50,
#                       help='Number of inferences to run')
#     parser.add_argument('--input-text', 
#                       default="Explain the theory of relativity in simple terms:",
#                       help='Input text for inference')
#     parser.add_argument('--max-new-tokens', type=int, default=100,
#                       help='Maximum number of tokens to generate')
    
#     args = parser.parse_args()
    
#     run_inference_workload(
#         model_name=args.model_name,
#         num_inferences=args.num_inferences,
#         input_text=args.input_text,
#         max_new_tokens=args.max_new_tokens,
#         output_dir=args.output_dir
#     )