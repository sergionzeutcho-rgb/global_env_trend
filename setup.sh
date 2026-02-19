mkdir -p ~/.streamlit/
echo "\
[theme]\n\
primaryColor = \"#2E7D32\"\n\
backgroundColor = \"#F1F8E9\"\n\
secondaryBackgroundColor = \"#DCEDC8\"\n\
textColor = \"#1B5E20\"\n\
font = \"sans serif\"\n\
\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
