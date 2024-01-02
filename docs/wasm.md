# LLM in your Browser - WebAssembly

Thanks to the new web assembly (WASM) technology and webGPU support of Chrome, it is now possible to run LLM in the local browser.
The newest deployment of biochatter (chatGSE-next) has also offer a WASM option. The following steps need to be taken to make the LLM run in your local docker set up.
1. `git clone https://github.com/xiaoranzhou/chatgse-next` (change to the biocypher repo after merge)
2. `git lfs install`  
`git clone https://huggingface.co/zxrzxr/zephyr-7b-beta-chatRDM-q4f32_1/ chatgse-next/chatgse/public/mistral`
3. `cd chatgse-next`
4. `docker-compose -f chatgse/docker-compose.yml up -d`
5. Open http://localhost:3000/#/webllm in **CHROME** (very important, other browser does not support webGPU yet)
6. Wait for loading of the LLM model, around 3-5 minutes. You might need to refresh the webpage until you see the "mistral" text (red circle):
   ![image](https://github.com/xiaoranzhou/biochatter/assets/29843510/684c735c-5d92-4cbe-9825-eb9eeec43bef)

7. Write questions in the chat input and click the send button behind the normal send button (blue circle).
