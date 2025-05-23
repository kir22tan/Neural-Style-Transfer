## 🗂️ Project Structure

```plaintext
neural-style-transfer/
├── notebooks/
│   └── neural_style_transfer.ipynb   # Main Colab notebook
│   └── collab                        # Google Collab link
├── presentation/
│   └── ppt.pptx                      
├── references/
│   └── Ppr1.pdf                      #by Leon A. Gatys
│   └── Ppr2.pdf                      #by Kapil Kashyap
├── src/
│   └── utils.py	                  #Image loading and displaying
│   └── model.py	                  #Loss functions, model builder, and optimization
│   └── main.py	                      #Run the pipeline: load images → transfer style → show output
├── requirements.txt                  # Python dependencies
├── results/                          
├── ABOUT.md                          # About the project
├── README.md  
└── ...
```

## 🚀 Installation

Option 1: On Google Colab

1. Open the Notebook  
   Click this link: [Google Colab Notebook](https://colab.research.google.com/)  

2. Upload Images  
   Upload your content and style images as instructed in the notebook.

3. Run All Cells  
   Go to Runtime > Run all to execute the entire workflow and generate your stylized image.

4. View Results  
   The output image will be displayed in the notebook and can be downloaded.


Option 2: Locally (if you have the code and dependencies)

1. Clone the Repository
```bash
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the Notebook  
   Launch Jupyter Notebook:
```bash
jupyter notebook
```
---
```plaintext
Also visit the ABOUT page for more details regarding the project.
```
---
