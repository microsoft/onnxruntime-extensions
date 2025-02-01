# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np

def read_file(file_path):
    """Reads a file and returns a list of lists with numbers from each row."""
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            # Split the line by commas and convert each value to float
            row = [float(x.strip()) for x in line.split(',')]
            data.append(row)
    return data

def get_mismatch_percentage(file1, file2, rtol=1e-03, atol=1e-02):
    """Compares the numbers in two files and checks the percentage of mismatched values."""
    # Read both files into lists of rows
    data1 = read_file(file1)
    data2 = read_file(file2)

    # Ensure both files have the same number of rows and columns
    if len(data1) != len(data2):
        print(f"Error: Files have different number of rows ({len(data1)} vs {len(data2)})")
        return
    
    # Flatten the data to compare all values at once
    flattened_data1 = np.array(data1).flatten()
    flattened_data2 = np.array(data2).flatten()

    # Check if both arrays have the same size
    if flattened_data1.size != flattened_data2.size:
        print(f"Error: Files have different number of total elements ({flattened_data1.size} vs {flattened_data2.size})")
        return
    
    # Compare using np.isclose() to check for close values within specified tolerances
    mismatched = ~np.isclose(flattened_data1, flattened_data2, rtol=rtol, atol=atol)
    
    # Count the number of mismatched values
    num_mismatched = np.sum(mismatched)
    
    # Check the percentage of mismatches
    mismatch_percentage = num_mismatched / flattened_data1.size

    print(f"Total mismatched values: {num_mismatched}")
    print(f"Mismatch percentage: {mismatch_percentage * 100:.2f}%")

    return mismatch_percentage

class TestPhi4AudioOutput(unittest.TestCase):

    def test_phi4_audio_output(self):
        # File paths

        # Note - these files are currently set to those with data from [:, :, :10] of the
        # PhiO audio processor output using the 'test/data/1272-141231-0002.wav' file

        file1_path = "test/data/models/phi-4/expected_output.txt"
        file2_path = "test/data/models/phi-4/actual_output.txt"

        # Compare the two files with a tolerance and mismatch threshold of 2% (HuggingFace's WhisperProcessor has around 1.5%)
        mismatch_percentage = get_mismatch_percentage(file1_path, file2_path)

        # For comparison, HuggingFace's WhisperProcessor has a mismatched percentage of around 1.5%.
        # With Phi4, ORT Extensions has around 0% mismatch with atol 1e-02.

        self.assertTrue(mismatch_percentage < 0.02)


if __name__ == "__main__":
    unittest.main()
