const { createApp, ref } = Vue

createApp({
    data () {
        return {
            message: 'Hello Vue!',
            context: {},
            count: 0,
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

            const chatWindow = document.getElementById('chat-window');
            
            // generate message box wrapper
            const boxWrapper = document.createElement('span');
            boxWrapper.classList.add('message-box-wrapper');
            
            // generate message box
            const messageBox = document.createElement('span');
            messageBox.classList.add('message-box');
            
            // decide which side
            if (side == 'l') {
                boxWrapper.classList.add('left')
            } else if (side == 'r') {
                boxWrapper.classList.add('right')
            }

            // add content
            messageBox.innerHTML = msg;
            
            boxWrapper.appendChild(messageBox);
            chatWindow.appendChild(boxWrapper);

            // scroll to bottom
            chatWindow.scrollTop = chatWindow.scrollHeight;

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
        async submit () {
            
            // skip if the submit is disabled 
            if (!this.submitEnabled) {
                return
            }

            const inputField = document.getElementById('msg-input-field');
            const payload = inputField.value;
            inputField.value = '';
            
            // build message box for chat window
            this.messageBox(payload, 'r');
            
            console.log('test 1')
            // prepare package for request
            let pkg = {
                sessionId: this.sessionId,
                message: payload,
                maxNewTokens: this.maxNewTokens
            }
            console.log('test 2')

            // request a response from API (this is most time consuming)
            const response = await this.request(pkg, '/')
            
            console.log('response', response)

            const answer = response.data.message;

            this.messageBox(answer, 'l');
            
        }

    },
    async mounted () {
        try {
            console.log('mounting.');
            // generate session id, a unique ientifier for server-side client
            this.sessionId = await this.generateSessionId();

            // this.request({'data':[1,2,3]}, '/');
            console.log('mounted.');
            this.messageBox('To make sense of it, try to remove minimum height & height -if it’s there, or give it height: 0 - on the parent, then add some height to this absolute-positioned child and see what happens. My guess now you will see the parent’s gone (u cant see it because it has no height) & u only see the child, or u see nothing at all - that is, if the parent is assigned with property', 'l');
            this.messageBox('This is another test', 'r');
            this.messageBox('This is another test', 'r')
            this.messageBox('This is another test', 'r')
            this.messageBox('This is another test', 'r')
            this.messageBox('This is another test', 'r')
            this.messageBox('To make sense of it, try to remove minimum height & height -if it’s there, or give it height: 0 - on the parent, then add some height to this absolute-positioned child and see what happens. My guess now you will see the parent’s gone (u cant see it because it has no height) & u only see the child, or u see nothing at all - that is, if the parent is assigned with property', 'l');

        } catch (error) {
            console.log('mount hook error:', error)
        }
    }
}).mount('#vue-app')