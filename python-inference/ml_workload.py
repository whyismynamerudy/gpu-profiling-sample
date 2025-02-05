import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import GPUtil
from datetime import datetime
import json
import os

class WorkloadProfiler:
    def __init__(self, output_dir="profiling_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/profiling_metrics_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filename}")
        
        # Calculate and print summary statistics
        summary = {
            "avg_gpu_utilization": sum(self.metrics["gpu_utilization"]) / len(self.metrics["gpu_utilization"]),
            "max_gpu_utilization": max(self.metrics["gpu_utilization"]),
            "avg_gpu_memory": sum(self.metrics["gpu_memory_used"]) / len(self.metrics["gpu_memory_used"]),
            "max_gpu_memory": max(self.metrics["gpu_memory_used"]),
            "avg_inference_time": sum(self.metrics["inference_times"]) / len(self.metrics["inference_times"]),
            "total_inferences": len(self.metrics["inference_times"])
        }
        
        summary_file = f"{self.output_dir}/summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

def run_inference_workload(
    model_name="facebook/opt-350m",  # Smaller model for testing, can be changed
    num_inferences=50,
    input_text="Explain the theory of relativity in simple terms:",
    max_new_tokens=100
):
    profiler = WorkloadProfiler()
    
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
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    print(f"Running {num_inferences} inferences...")
    for i in range(num_inferences):
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
        profiler.metrics["inference_times"].append(inference_time)
        
        if i % 10 == 0:  # Progress update every 10 inferences
            print(f"Completed {i+1}/{num_inferences} inferences")
    
    profiler.save_metrics()
    print("Workload completed!")

if __name__ == "__main__":
    # You can modify these parameters as needed
    run_inference_workload(
        model_name="facebook/opt-350m",  # Can be changed to larger models
        num_inferences=50,
        input_text="Explain the theory of relativity in simple terms:",
        max_new_tokens=100
    )