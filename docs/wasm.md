# LLM in your Browser - WebAssembly

Thanks to the new web assembly (WASM) technology and webGPU support of Chrome, it is now possible to run LLM in the local browser.
The newest deployment of biochatter (chatGSE-next) has also offer a WASM option. The following steps need to be taken to make the LLM run in your local docker set up.
1. `git clone https://github.com/xiaoranzhou/chatgse-next` (change to the biocypher repo after merge)
2. `git lfs install`  
`git clone https://huggingface.co/zxrzxr/zephyr-7b-beta-chatRDM-q4f32_1/chatgse-next/chatgse/public/mistral`
3. `docker-compose -f chatgse/docker-compose.yml up -d`
4. Open http://localhost:3000/#/webllm in **CHROME** (very important, other browser does not support webGPU yet)
5. Wait for loading of the LLM model, around 3-5 minutes.
6. Write questions in the chat input and click the send button behind the normal send button.
