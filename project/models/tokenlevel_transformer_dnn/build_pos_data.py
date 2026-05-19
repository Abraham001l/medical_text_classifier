import os
import glob
import json
import csv
from transformers import AutoTokenizer

def build_pos_data():
    # 1. Initialize the Clinical Longformer Tokenizer
    # We use the fast tokenizer to take advantage of 'return_offsets_mapping'
    model_name = "yikuan8/Clinical-Longformer"
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # 2. Define the file paths based on your directory structure
    inpatient_path = "../../data/with_text/gold/Inpatient/ICD-10/1.0/*.json"
    profee_path = "../../data/with_text/gold/Profee/ICD-10/1.0/*.json"
    
    # Use glob to find all matching JSON files
    file_paths = glob.glob(inpatient_path) + glob.glob(profee_path)
    print(f"Found {len(file_paths)} JSON files to process.")
    
    # 3. Setup the output CSV
    output_file = "pos_data.csv"
    
    # Open the CSV file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the header exactly as specified
        writer.writerow(["token_char_rep_list", "token_embd_rep_list", "token_classif_list"])
        
        # 4. Iterate over every found JSON file
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as jf:
                try:
                    data = json.load(jf)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON in {fp}. Skipping.")
                    continue
                
            # Iterate through all notes in the current JSON file
            for note in data.get("notes", []):
                text = note.get("text", "")
                
                # Skip empty notes
                if not text.strip():
                    continue
                    
                annotations = note.get("annotations", [])
                
                # 5. Tokenize the text while capturing character offsets
                # We truncate to 4096 because that is the Longformer's max capacity
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=4096,
                    return_offsets_mapping=True 
                )
                
                input_ids = encoding["input_ids"]
                offsets = encoding["offset_mapping"]
                
                # Convert the IDs back into readable string representations
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                
                # Initialize the classification list with all 0s (Negative)
                labels = [0] * len(input_ids)
                
                # Extract all evidence spans from the MDACE annotations
                evidence_spans = []
                for ann in annotations:
                    begin = ann.get("begin", -1)
                    end = ann.get("end", -1)
                    if begin != -1 and end != -1:
                        evidence_spans.append((begin, end))
                        
                # 6. Map annotations to tokens
                for i, (tok_start, tok_end) in enumerate(offsets):
                    # Special tokens (like <s>, </s>, <pad>) map to (0, 0)
                    # We leave their classification as 0
                    if tok_start == tok_end:  
                        continue
                    
                    is_codable = 0
                    for (ev_start, ev_end) in evidence_spans:
                        # A token is marked as positive (1) if it overlaps with an evidence span
                        # Logic: Token starts before evidence ends AND Token ends after evidence starts
                        if tok_start < ev_end and tok_end > ev_start:
                            is_codable = 1
                            break # No need to check other spans for this token once it's flagged
                            
                    labels[i] = is_codable
                    
                # 7. Write the extracted lists to the CSV row as JSON strings
                writer.writerow([
                    json.dumps(tokens),
                    json.dumps(input_ids),
                    json.dumps(labels)
                ])
                
    print(f"Data successfully compiled and saved to {output_file}.")

if __name__ == "__main__":
    build_pos_data()