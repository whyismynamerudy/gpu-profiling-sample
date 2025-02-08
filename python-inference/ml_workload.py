import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import GPUtil
from datetime import datetime
import json
import os
import argparse

class WorkloadProfiler:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            "timestamps": [],
            "gpu_utilization": [],
            "gpu_memory_used": [],
            "cpu_utilization": [],
            "system_memory_used": [],
            "inference_times": []
        }
        
    def _collect_metrics(self):
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Using first GPU
            self.metrics["gpu_utilization"].append(gpu.load * 100)
            self.metrics["gpu_memory_used"].append(gpu.memoryUsed)
        else:
            self.metrics["gpu_utilization"].append(0)
            self.metrics["gpu_memory_used"].append(0)
            
        self.metrics["cpu_utilization"].append(psutil.cpu_percent())
        self.metrics["system_memory_used"].append(psutil.virtual_memory().percent)
        self.metrics["timestamps"].append(datetime.now().isoformat())

    def save_metrics(self):
        # Save detailed metrics
        metrics_file = os.path.join(self.output_dir, f"ml_metrics_{self.run_timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Detailed metrics saved to {metrics_file}")
        
        # Calculate and save summary statistics
        summary = {
            "avg_gpu_utilization": sum(self.metrics["gpu_utilization"]) / len(self.metrics["gpu_utilization"]),
            "max_gpu_utilization": max(self.metrics["gpu_utilization"]),
            "avg_gpu_memory": sum(self.metrics["gpu_memory_used"]) / len(self.metrics["gpu_memory_used"]),
            "max_gpu_memory": max(self.metrics["gpu_memory_used"]),
            "avg_inference_time": sum(self.metrics["inference_times"]) / len(self.metrics["inference_times"]),
            "total_inferences": len(self.metrics["inference_times"]),
            "timestamp": self.run_timestamp
        }
        
        summary_file = os.path.join(self.output_dir, f"ml_summary_{self.run_timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

def run_inference_workload(
    model_name="facebook/opt-350m",
    num_inferences=50,
    input_text="Explain the theory of relativity in simple terms:",
    max_new_tokens=100,
    output_dir="/app/profiling_results"
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    profiler = WorkloadProfiler(output_dir)
    
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    print(f"Running {num_inferences} inferences...")
    for i in range(num_inferences):
        profiler._collect_metrics()
        
        inference_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        inference_time = time.time() - inference_start
        profiler.metrics["inference_times"].append(inference_time)
        
        if i % 10 == 0:
            print(f"Completed {i+1}/{num_inferences} inferences")
    
    profiler.save_metrics()
    print("Workload completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ML inference workload')
    parser.add_argument('--output-dir', required=True,
                      help='Directory to save profiling results')
    parser.add_argument('--model-name', default="facebook/opt-350m",
                      help='Model to use for inference')
    parser.add_argument('--num-inferences', type=int, default=50,
                      help='Number of inferences to run')
    parser.add_argument('--input-text', 
                      default="Explain the theory of relativity in simple terms:",
                      help='Input text for inference')
    parser.add_argument('--max-new-tokens', type=int, default=100,
                      help='Maximum number of tokens to generate')
    
    args = parser.parse_args()
    
    run_inference_workload(
        model_name=args.model_name,
        num_inferences=args.num_inferences,
        input_text=args.input_text,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )