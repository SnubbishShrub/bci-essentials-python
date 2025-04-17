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
# print(f"Sample rate: {sample_rate} Hz")
print(f"Channel count: {n_channels}")


def original_pull_chunk():
    # Read data in chunks
    chunk_size = 128  # Adjust this value based on your needs
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
                if len(chunk_timestamps) > 1:
                    time_diffs = np.diff(chunk_timestamps)
                    max_diff = np.max(time_diffs)
                    if max_diff > threshold:
                        print(f"Warning: Large timing gap detected: {max_diff:.6f} seconds")
                        
                # Check timing difference between chunks
                if len(all_timestamps) > len(chunk_timestamps):
                    chunk_gap = chunk_timestamps[0] - all_timestamps[-len(chunk_timestamps)-1]
                    if chunk_gap > threshold:
                        print(f"Warning: Large gap between chunks: {chunk_gap:.6f} seconds")
                        
        except KeyboardInterrupt:
            print("\nStream reading with traditional method stopped.")
            break

def np_pull_chunk():
    # Initialize buffer and timestamps
    buffer = np.empty((sample_rate//2, n_channels), dtype=np.float32)
    timestamps = np.empty(sample_rate//2, dtype=np.float64)
    current_idx = 0
    all_timestamps = []  # Store all timestamps

    while True:
        try: 
            # Pull chunk into buffer starting at current_idx
            inlet.pull_chunk(
                max_samples=buffer.shape[0]-current_idx,
                dest_obj=buffer[current_idx:]
            )

            # Save time stamps
            all_timestamps.append(buffer)

            
            if buffer.size > 0:
                # Store timestamps
                timestamps[current_idx:current_idx+len(chunk_timestamps)] = chunk_timestamps
                current_idx += len(chunk_timestamps)
                all_timestamps.extend(chunk_timestamps)
                
                # Resize buffers if full
                if current_idx >= buffer.shape[0]:
                    new_size = buffer.shape[0] * 2
                    # Resize data buffer
                    new_buffer = np.empty((new_size, n_channels), dtype=np.float32)
                    new_buffer[:buffer.shape[0]] = buffer
                    buffer = new_buffer
                    # Resize timestamps buffer
                    new_timestamps = np.empty(new_size, dtype=np.float64)
                    new_timestamps[:timestamps.shape[0]] = timestamps
                    timestamps = new_timestamps
                    
            # Process valid data slice
            if current_idx > 0:
                check_timing(timestamps[:current_idx], threshold, all_timestamps)

        except KeyboardInterrupt:
            print("\nStream reading with numpy method stopped.")
            break

def check_timing(timestamps, threshold, all_timestamps):
    scale = 1e3  # Scale timestamps to milliseconds for better readability
    # Check timing differences within this chunk
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        max_diff = np.max(time_diffs)
        if max_diff > threshold:
            print(f"Warning: Gap WITHIN chunk detected: {scale*max_diff:.2f} msec")
            
    # Check timing difference between chunks
    if len(all_timestamps) > len(timestamps):
        chunk_gap = timestamps[0] - all_timestamps[-len(timestamps)-1]
        if chunk_gap > threshold:
            print(f"Warning: Gap BETWEEN chunks detected: {scale*chunk_gap:.2f} msec")


methods = {
    "1": original_pull_chunk(),
    "2": np_pull_chunk(),
}
method = input("Enter method number: ")
if method in methods:
    methods[method]