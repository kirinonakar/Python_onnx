import os
import torch
from safetensors.torch import save_file
import argparse
import traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def convert_pth_to_safetensors(input_path, output_path=None, status_callback=None):
    """
    Converts a PyTorch .pth checkpoint to a .safetensors file.
    """
    try:
        if not os.path.exists(input_path):
            error_msg = f"Error: File {input_path} not found."
            if status_callback: status_callback(error_msg, "red")
            print(error_msg)
            return False

        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".safetensors"
        
        msg = f"[*] Loading PyTorch checkpoint: {os.path.basename(input_path)}"
        if status_callback: status_callback(msg, "blue")
        print(msg)

        # Load on CPU to avoid requiring a GPU for conversion
        state_dict = torch.load(input_path, map_location="cpu")
        
        # Handle common nesting patterns (BasicSR, PyTorch Lightning, etc.)
        if isinstance(state_dict, dict):
            for key in ["params", "state_dict", "model", "params_ema"]:
                if key in state_dict:
                    msg = f"[i] Found '{key}' key, extracting..."
                    if status_callback: status_callback(msg, "blue")
                    print(msg)
                    state_dict = state_dict[key]
                    break
        
        # Ensure it's a dict and all values are tensors
        if not isinstance(state_dict, dict):
            error_msg = f"[!] Error: Loaded object is not a dictionary (type: {type(state_dict)})."
            if status_callback: status_callback(error_msg, "red")
            print(error_msg)
            return False

        # Check if all items are tensors; if not, filter them out
        new_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                new_state_dict[k] = v.contiguous()
            else:
                print(f"[w] Warning: Skipping non-tensor key '{k}' (type: {type(v)})")
                
        if not new_state_dict:
            error_msg = "[!] Error: No valid tensors found in the checkpoint."
            if status_callback: status_callback(error_msg, "red")
            print(error_msg)
            return False

        msg = f"[*] Saving to Safetensors: {os.path.basename(output_path)}"
        if status_callback: status_callback(msg, "blue")
        print(msg)

        save_file(new_state_dict, output_path)
        
        success_msg = "[+] Conversion successful!"
        if status_callback: status_callback(success_msg, "green")
        print(success_msg)
        return True

    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        if status_callback: status_callback(error_msg, "red")
        print(traceback.format_exc())
        return False

class SafetensorsConverterGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PTH to Safetensors Converter")
        self.geometry("600x400")
        self.grid_columnconfigure(0, weight=1)

        # Header
        self.header_frame = ctk.CTkFrame(self, height=80, corner_radius=0, fg_color="#1f2937")
        self.header_frame.pack(fill="x", pady=(0, 20))
        
        self.label = ctk.CTkLabel(self.header_frame, text="🔒 PTH to Safetensors ✨", font=("Arial", 24, "bold"), text_color="#60a5fa")
        self.label.pack(pady=20)

        # Main Container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=40)

        # 1. Input File Selection
        self.input_label = ctk.CTkLabel(self.main_container, text="Input Model (.pth)", font=("Arial", 14, "bold"))
        self.input_label.pack(anchor="w", pady=(10, 5))
        
        self.input_frame = ctk.CTkFrame(self.main_container, fg_color="#374151")
        self.input_frame.pack(fill="x", pady=5)

        self.input_path_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Path to .pth file...", border_width=0, height=35)
        self.input_path_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")

        self.browse_input_btn = ctk.CTkButton(self.input_frame, text="Browse", command=self.browse_input, width=100, fg_color="#3b82f6", hover_color="#2563eb")
        self.browse_input_btn.pack(side="right", padx=10, pady=10)

        # 2. Output Path (Optional)
        self.output_label = ctk.CTkLabel(self.main_container, text="Output Path (Optional)", font=("Arial", 14, "bold"))
        self.output_label.pack(anchor="w", pady=(10, 5))
        
        self.output_frame = ctk.CTkFrame(self.main_container, fg_color="#374151")
        self.output_frame.pack(fill="x", pady=5)

        self.output_path_entry = ctk.CTkEntry(self.output_frame, placeholder_text="Defaults to input filename...", border_width=0, height=35)
        self.output_path_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")

        self.browse_output_btn = ctk.CTkButton(self.output_frame, text="Save As", command=self.browse_output, width=100, fg_color="#374151", hover_color="#4b5563")
        self.browse_output_btn.pack(side="right", padx=10, pady=10)

        # 3. Convert Button
        self.convert_btn = ctk.CTkButton(self, text="🚀 START CONVERSION", command=self.start_conversion, height=50, font=("Arial", 18, "bold"), fg_color="#10b981", hover_color="#059669")
        self.convert_btn.pack(pady=30, padx=40, fill="x")

        # Status Footer
        self.status_footer = ctk.CTkFrame(self, height=40, fg_color="#111827", corner_radius=0)
        self.status_footer.pack(side="bottom", fill="x")
        
        self.status_label = ctk.CTkLabel(self.status_footer, text="Ready", text_color="#9ca3af", font=("Arial", 12))
        self.status_label.pack(pady=5)

    def browse_input(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Weight", "*.pth")])
        if file_path:
            self.input_path_entry.delete(0, "end")
            self.input_path_entry.insert(0, file_path)
            
            # Auto-fill output if empty
            if not self.output_path_entry.get():
                out_path = os.path.splitext(file_path)[0] + ".safetensors"
                self.output_path_entry.insert(0, out_path)

    def browse_output(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".safetensors", filetypes=[("Safetensors", "*.safetensors")])
        if file_path:
            self.output_path_entry.delete(0, "end")
            self.output_path_entry.insert(0, file_path)

    def update_status(self, text, color_theme="blue"):
        colors = {
            "blue": "#60a5fa",
            "green": "#34d399",
            "red": "#f87171",
            "yellow": "#fbbf24"
        }
        color = colors.get(color_theme, "#9ca3af")
        self.status_label.configure(text=text, text_color=color)
        self.update()

    def start_conversion(self):
        input_path = self.input_path_entry.get().strip()
        output_path = self.output_path_entry.get().strip() or None

        if not input_path:
            messagebox.showerror("Error", "Please select an input .pth file.")
            return

        self.convert_btn.configure(state="disabled")
        
        success = convert_pth_to_safetensors(
            input_path, 
            output_path, 
            status_callback=self.update_status
        )

        if success:
            messagebox.showinfo("Success", "Model converted successfully!")
            self.update_status("Conversion Completed", "green")
        else:
            messagebox.showerror("Error", "Conversion failed. Check console for details.")
            self.update_status("Conversion Failed", "red")
            
        self.convert_btn.configure(state="normal")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .pth files to .safetensors format.")
    parser.add_argument("input", nargs="?", help="Path to the input .pth file")
    parser.add_argument("--output", "-o", help="Path to the output .safetensors file (optional)", default=None)
    
    args = parser.parse_args()
    
    if args.input:
        # CLI Mode
        input_file = os.path.abspath(args.input)
        output_file = os.path.abspath(args.output) if args.output else None
        convert_pth_to_safetensors(input_file, output_file)
    else:
        # GUI Mode
        app = SafetensorsConverterGUI()
        app.mainloop()
