import numpy as np
from pylsl import StreamInlet, StreamInfo, resolve_byprop, FOREVER

# Detect streams
print("Looking for streams...")
streams = resolve_byprop('type', 'EEG', timeout=2.0)

# Let user select stream
print("\nSelect stream to read from:")
for i, stream in enumerate(streams):
    print(f"{i}: {stream.name()}")
    
selection = int(input("Enter stream number: "))
selected_stream = streams[selection]
print(f"\nSelected stream: {selected_stream.name()}")

# Read data by chunks
print("\nReading data...")
inlet = StreamInlet(selected_stream)
sample_rate = 256
n_channels = inlet.channel_count
threshold = 2/sample_rate  # Maximum allowed time difference
chunk_size = sample_rate // 16  # Number of samples to read at once
# print(f"Sample rate: {sample_rate} Hz")
print(f"Channel count: {n_channels}")


def original_pull_chunk():
    # Read data in chunks
    all_timestamps = []  # Store all timestamps
    print(f"Timing threshold: {threshold:.6f} seconds")

    while True:
        try:
            # get_data() returns tuple (samples, timestamps)
            samples, chunk_timestamps = inlet.pull_chunk(max_samples=chunk_size, timeout=0)
            
            if len(chunk_timestamps)>0:
                # chunk_data = np.array(samples)
                # print(f"Got chunk of {len(chunk_timestamps)} samples")
                # print(f"Chunk shape: {chunk_data.shape} (samples × channels)")
                
                
                # Add new timestamps to the list
                all_timestamps.extend(chunk_timestamps)
                
                # Check timing differences within this chunk
                check_timing(chunk_timestamps, all_timestamps, threshold)                    
                        
        except KeyboardInterrupt:
            print("\nStream reading with traditional method stopped.")
            break

def np_pull_chunk():
    # Initialize fixed buffer and timestamps
    buffer = np.empty((chunk_size, n_channels), dtype=np.float32)
    all_timestamps = []  # Store all timestamps
    all_samples = []     # Store all samples

    while True:
        try: 
            # Pull chunk into buffer from start (overwrite)
            _, chunk_timestamps = inlet.pull_chunk(
                max_samples=chunk_size,
                dest_obj=buffer,
                timeout=0.001
            )

            if len(chunk_timestamps) > 0:
                # Store valid portion of data
                valid_data = buffer.copy()[:len(chunk_timestamps)]
                all_samples.append(valid_data)
                all_timestamps.extend(chunk_timestamps)
                
                # Process valid data slice (use chunk_timestamps directly)
                check_timing(chunk_timestamps, all_timestamps, threshold)

        except KeyboardInterrupt:
            # Convert all samples to single numpy array
            print(f"\nStream reading stopped")
            break


def check_timing(chunk_timestamps, all_timestamps, threshold):
    scale = 1e3  # Scale timestamps to milliseconds for better readability
    # Check timing differences within this chunk
    if len(chunk_timestamps) > 1:
        time_diffs = np.diff(chunk_timestamps)
        max_diff = np.max(time_diffs)
        if max_diff > threshold:
            print(f"Warning: Gap WITHIN chunk detected: {scale*max_diff:.2f} msec, {1/max_diff:.2f} Hz")
            
    # Check timing difference between chunks
    if len(all_timestamps) > len(chunk_timestamps):
        chunk_gap = chunk_timestamps[0] - all_timestamps[-len(chunk_timestamps)-1]
        if chunk_gap > threshold:
            print(f"Warning: Gap BETWEEN chunks detected: {scale*chunk_gap:.2f} msec, {1/chunk_gap:.2f} Hz")


methods = {
    "1": original_pull_chunk,
    "2": np_pull_chunk,
}
print("\nSelect method to use:")
print("1: Original pull chunk method")
print("2: Numpy optimized pull chunk method")
method = input("Enter method number: ")
if method in methods:
    methods[method]()