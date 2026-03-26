import time
from datetime import datetime

def monitor_output_folder(output_paths, stop_event, total_files):
    """
    Monitors the generation of expected output files and periodically prints progress with ETA.

    Args:
        output_paths (list): List of expected output file Paths (one per case for p.exists() tracking).
        stop_event (threading.Event): Event to signal when the main process is finished.
        total_files (int): Total number of files to process (should match len(output_paths)).
    """
    # Initialize counters
    processed_count_last = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        # Count currently processed files
        processed_count = sum(1 for p in output_paths if p.exists())
        
        # Display progress if new files are detected
        if processed_count > processed_count_last:
            # Calculate progress percentage
            progress = (processed_count / total_files) * 100
            
            # Calculate timing information
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            
            # Calculate Estimated Time Remaining (ETR)
            if processed_count > 0:
                time_per_file = elapsed / processed_count
                remaining_files = total_files - processed_count
                remaining_time = time_per_file * remaining_files
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                
                # Estimate Time of Arrival (ETA)
                eta_timestamp = datetime.now().timestamp() + remaining_time
                eta_str = datetime.fromtimestamp(eta_timestamp).strftime("%H:%M:%S")
                
                # Calculate processing speed
                newly_processed = processed_count - processed_count_last
                # Logic for current interval speed
                time_since_last_update = time.time() - (start_time + (processed_count_last * time_per_file))
                if time_since_last_update > 0:
                    speed = newly_processed / time_since_last_update
                    speed_str = f"{speed:.2f} files/sec"
                else:
                    speed_str = "Calculating..."
            else:
                remaining_str = "Calculating..."
                eta_str = "Calculating..."
                speed_str = "Calculating..."
            
            # Create visual progress bar
            bar_length = 30
            filled_length = int(bar_length * processed_count / total_files)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Print progress dashboard
            print(f"\n{'='*80}")
            print(f"🔄 Prediction Progress: {processed_count}/{total_files} ({progress:.1f}%)")
            print(f"[{bar}]")
            print(f"  - Processed: {processed_count} files")
            print(f"  - Elapsed Time: {elapsed_str}")
            print(f"  - Estimated Remaining: {remaining_str}")
            print(f"  - Estimated Completion (ETA): {eta_str}")
            print(f"  - Current Speed: {speed_str}")
            print(f"{'='*80}\n")
            
            # Update last processed count
            processed_count_last = processed_count
        
        # Polling interval (wait 5 seconds before next check)
        time.sleep(5)
