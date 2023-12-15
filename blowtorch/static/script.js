const { createApp, ref } = Vue

const app = createApp({
    data () {
        return {
            message: 'Hello Vue!',
            context: {},
            count: 0,
            inputFillSpeed: 1.2,
            maxNewTokens: 128,
            sessionId: null,
            
            submitEnabled: true
        }
    },
    methods: {

        async generateSessionId() {

            /*
            Generates a unique session identifier.
            */

            // generate seed
            let seed = '';
            let characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
            for (let i = 0; i < 64; i++) {
                seed += characters.charAt(Math.floor(Math.random() * characters.length));
            }

            console.log('seed', seed)

            // hash
            const msgUint8 = new TextEncoder().encode(seed);                           // encode as (utf-8) Uint8Array
            const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);           // hash the message
            const hashArray = Array.from(new Uint8Array(hashBuffer));                     // convert buffer to byte array
            const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join(''); // convert bytes to hex string
            console.log('hash', hashHex);
            return hashHex

        },
        incrementCount() {
            this.count++
        },
        messageBox (msg, side) {

            /* 
            Creates a message box in chat window.
            'side' determines the side 'l' or 'r' 
            for left of right, resp. 
            */

            const chatWindow = document.getElementById('chat-window');
            
            // generate message box wrapper
            const boxWrapper = document.createElement('span');
            boxWrapper.classList.add('message-box-wrapper');
            
            // generate message box
            const messageBox = document.createElement('span');
            messageBox.classList.add('message-box');
            
            // decide which side
            if (side == 'l') {
                boxWrapper.classList.add('left');
                messageBox.classList.add('gray')
            } else if (side == 'r') {
                boxWrapper.classList.add('right');
                messageBox.classList.add('green')
            }

            // add content
            messageBox.innerHTML = msg;
            
            boxWrapper.appendChild(messageBox);
            chatWindow.appendChild(boxWrapper);

            // scroll to bottom
            this.scrollToBottom();

            return messageBox
            
        },
        request (options, path, json=true) {
            return new Promise(function (resolve, reject) {
                var xhr = new XMLHttpRequest(); 
                xhr.open("POST", path, true); 
                if (json) {
                    xhr.setRequestHeader("Content-type", "application/json;charset=UTF-8"); 
                }
                xhr.onreadystatechange = function () {  
                    if (xhr.readyState == 4 && xhr.status == 200) {
                        var json = JSON.parse(xhr.responseText);
                        if (Object.keys(json).includes('errors') && json['errors'].length != 0) { // if errors occur
                            console.log('server:', json['errors'])
                        } resolve(json);
                    }
                }
                xhr.onerror = function(e) {
                    reject({'errors': ['error during request: no connection']})
                }
                xhr.send(JSON.stringify(options)); 
            });
        },
        async scrollToBottom () {
            /*
            Scrolls chat window to bottom.
            */
            let chatWindow = document.getElementById('chat-window');
            chatWindow.scrollTop = chatWindow.scrollHeight;
        },
        sleep: function (seconds) {
            return new Promise(function(resolve) {
                setTimeout(function() {
                    resolve(0);
                }, 1000*seconds);
            });
        },
        async submit () {

            /*
            Submits message to server and awaits response.
            Will create message boxes in chat window appropriately.
            */
            
            // skip if the submit is disabled 
            if (!this.submitEnabled) {
                return
            }

            // extract payload from input field
            const inputField = document.getElementById('msg-input-field');
            const payload = inputField.value;
            inputField.value = '';
            
            // build message box for chat window
            this.messageBox(payload, 'r');
            
            // prepare package for request
            let pkg = {
                sessionId: this.sessionId,
                message: payload,
                maxNewTokens: this.maxNewTokens
            }

            // disable submit function
            this.submitEnabled = false;

            // spawn an empty message box
            const msgBox = this.messageBox('', 'l');

            // add a custom event listener which scrolls chat to bottom
            // when a message box 
            let resizeObserver = new ResizeObserver(entries => {
                this.scrollToBottom();
            });
            resizeObserver.observe(msgBox);
            
            // make a wait animation in message box
            //this.waitAnimation(msgBox); // will terminate whe 

            // request a response from API (this is most time consuming)
            const response = await this.request(pkg, '/')

            // enable submit functionality again
            this.submitEnabled = true;
            
            console.log('response', response)
            // derive answer payload
            const answer = response.data.message;
            
            // fill into message box
            msgBox.innerHTML = '';
            ind = 0
            for (let i = 0; i < answer.length; i++) {
                const char = answer[i];
                msgBox.innerHTML += char;
                await this.sleep(.1 / this.inputFillSpeed);
            }

            // stop event listener for message box
            resizeObserver.unobserve(msgBox);
            
        }

    },
    async mounted () {

        try {

            console.log('mounting.');

            // generate session id, a unique ientifier for server-side client
            this.sessionId = await this.generateSessionId();

            const inputField = document.getElementById('msg-input-field')

            // bind send button to ENTER key
            let submitFunction = this.submit;
            inputField.addEventListener('keydown', function(event) {
                // Check if the key pressed was 'Enter'
                if (event.key === 'Enter') {
                    console.log('submit ...');
                    submitFunction()
                }
            });

            // this.request({'data':[1,2,3]}, '/');
            console.log('mounted.');
            this.messageBox("I am an AI assistant, how may I help you?", 'l');

        } catch (error) {
            console.log('mount hook error:', error)
        }
    },
    async waitAnimation (messageBox) {
        while (!this.submitEnabled) {
            if (messageBox.innerHTML == '...'){
                messageBox.innerHTML = ''
            }
            messageBox.innerHTML += '.';
            await this.sleep(.05)
        }
    }
})

app.mount('#vue-app')