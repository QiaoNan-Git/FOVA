import numpy as np
import random
from offlinerlkit.buffer import ReplayBuffer


def sample(buffer: ReplayBuffer, data_size: int, local_data_size: int):
    """
    Sample a batch of transitions from the buffer.
    
    Args:
        buffer: ReplayBuffer instance
        data_size: total size of the dataset
        local_data_size: number of samples to draw
        
    Returns:
        Batch of sampled transitions
    """
    length = data_size  # len(dataset["observations"])
    assert 1 <= local_data_size
    
    indices = np.random.randint(0, length, local_data_size)
    return buffer[indices]


def collectTrajs(start_list: list, end_list: list, local_data_size: int):
    """
    Collect trajectories from start and end indices.
    
    Args:
        start_list: list of trajectory start indices
        end_list: list of trajectory end indices
        local_data_size: target total length of collected trajectories
        
    Returns:
        collected_trajs: list of (start, end) tuples
        total_length_list: list of lengths for each collected trajectory
    """
    # Calculate trajectory lengths
    traj_lengths = [end - start + 1 for start, end in zip(start_list, end_list)]
    
    # Combine start positions and trajectory lengths
    trajs = list(zip(start_list, traj_lengths))
    
    # Random shuffle trajectories (optional)
    # random.shuffle(trajs)
    
    # Variables for collecting results
    collected_trajs = []
    total_length = 0
    total_length_list = []
    
    # Generate random start index
    if end_list[-1] > local_data_size:
        random_start = random.randint(1, end_list[-1] - local_data_size)
    else:
        random_start = 1
    
    # Find the first trajectory index where start > random_start
    random_index = next((i for i, (start, _) in enumerate(trajs) if start > random_start), None)
    
    # Process trajectories from random index
    for start, length in trajs[random_index:]:
        if total_length + length > local_data_size:
            # Check if we need to truncate the last trajectory
            remaining_length = local_data_size - total_length
            if remaining_length > 50:  # Skip if too short
                collected_trajs.append((start, start + remaining_length - 1))
                total_length_list.append(remaining_length)
                total_length += remaining_length
            break
        collected_trajs.append((start, start + length - 1))
        total_length_list.append(length)
        total_length += length
    
    return collected_trajs, total_length_list


def extract_and_combine_trajs(dataset: dict, collected_trajs: list):
    """
    Extract and combine trajectory segments from dataset.
    
    Args:
        dataset: original dataset dictionary
        collected_trajs: list of (start, end) tuples
        
    Returns:
        new_dataset: combined dataset dictionary
    """
    # Create a new dictionary to store extracted data
    new_dataset = {key: [] for key in dataset.keys()}
    
    # Iterate over all collected trajectory intervals
    for start, end in collected_trajs:
        for key in dataset.keys():
            # Extract trajectory segment from each key's array
            segment = dataset[key][start:end + 1]  # +1 because end is inclusive
            # Add segment to the new list
            new_dataset[key].append(segment)
    
    # Concatenate all arrays along the first axis
    for key in new_dataset.keys():
        new_dataset[key] = np.concatenate(new_dataset[key], axis=0)
    
    return new_dataset
