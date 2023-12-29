const { createApp, ref } = Vue

const app = createApp({
    data () {
        return {
            message: 'Hello Vue!',
            context: {},
            count: 0,
            inputFillSpeed: 5,
            sessionId: null,
            settings: { 
                maxNewTokens: 256,
                contextLength: 6000,
                typeWriterMode: true
            },
            submitEnabled: true
        }
    },
    methods: {
        async formatMessage (message) {
            /*
            Formats a message into hmtl compliant format.
            */
            return message
            // formatted = message.split('\n').join('<br>');

            // return '<p>'+formatted+'</p>'
        },
        async generateSessionId () {

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
        incrementCount () {
            this.count++
        },
        identifyLanguage(string) {

            /* Determines the programming language which is contained in a string.
            If no code is detected the return is null. */

            var python1 = new RegExp('.*from .* import .*'),
                python2 = new RegExp('.*/env python3.*'),
                python3 = new RegExp('.*print(.*).*'),
                js1 = new RegExp('.*function .*\\(.*\\).*'),
                js2 = new RegExp('.*console\\.log\\(.*\\).*'),
                c_cpp = new RegExp('.*#include <.*>.*'),
                cuda = new RegExp('.*__global__ void .*'),
                rust = new RegExp('.*fn main\\(\\) {.*'),
                golang = new RegExp('.*func main\\(\\) {.*'),
                html = new RegExp('.*<html>.*'),
                typescript = new RegExp('.*let .*: .*;.*'),
                bash = new RegExp('.*#!\\/bin\\/bash.*'),
                powershell = new RegExp('.*Write-Host .*'),
                r = new RegExp('.*<-.*'),
                ruby = new RegExp('.*def .*end.*'),
                perl = new RegExp('.*use strict;.*');
                unknown = new RegExp('.*```.*');
            if (string.match(python1) || string.match(python2) || string.match(python3)) {
                return 'python';
            } else if (string.match(js1) || string.match(js2)) {
                return 'javascript';
            } else if (string.match(c_cpp)) {
                return 'c/c++';
            } else if (string.match(cuda)) {
                return 'cuda';
            } else if (string.match(rust)) {
                return 'rust';
            } else if (string.match(golang)) {
                return 'golang';
            } else if (string.match(html)) {
                return 'html';
            } else if (string.match(typescript)) {
                return 'typescript';
            } else if (string.match(bash)) {
                return 'bash';
            } else if (string.match(powershell)) {
                return 'powershell';
            } else if (string.match(r)) {
                return 'R';
            } else if (string.match(ruby)) {
                return 'ruby';
            } else if (string.match(perl)) {
                return 'perl';
            } else if (string.match(unknown)) {
                return 'unknown';
            } else {
                return null;
            }

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
        sleep (seconds) {
            return new Promise(function(resolve) {
                setTimeout(function() {
                    resolve(0);
                }, 1000*seconds);
            });
        },
        async render (message, msgBox) {

            // split paragraphs
            const paragraphs = message.split('\n');

            var code = '';

            codeStarted = false;

            for (let paragraph of paragraphs) {

                // check for coding block
                if (code || this.identifyLanguage(paragraph)) {
                    if (paragraph.includes('```') && codeStarted) {
                        code += paragraph.replace('```', '');
                        paragraph = code;
                    } else {
                        codeStarted = true;
                        code += paragraph;
                        continue
                    }
                }

                // create new paragraph element and append to messagebox
                p = document.createElement('p');
                msgBox.appendChild(p);
                
                // init code block if detected
                if (code) {
                    codeStarted = false;
                    p.classList.add('source-code');
                    code = ''
                }

                // insert the text to p element
                if (this.settings.typeWriterMode) {
                    p.textContent = '';
                    for (let i = 0; i < paragraph.length; i++) {
                        p.textContent += paragraph[i];
                        await this.sleep(.1 / this.inputFillSpeed);
                    }
                } else {
                    p.textContent = paragraph;
                }
                
            }

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
                maxNewTokens: this.settings.maxNewTokens,
                contextLength: this.settings.contextLength
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
            this.waitAnimation(msgBox); // will terminate when submit is enabled

            // request a response from API (this is most time consuming)
            const response = await this.request(pkg, '/')

            // enable submit functionality again
            this.submitEnabled = true;
            
            console.log('response', response)
            // derive answer payload
            const answer = response.data.message;
            const formattedAnswer = await this.formatMessage(answer);
            
            // render message
            await this.render(formattedAnswer, msgBox);
            
            // stop event listener for message box
            resizeObserver.unobserve(msgBox);
            
        },
        async waitAnimation (messageBox) {
            console.log('ani 0')
            while (!this.submitEnabled) {
                console.log('ani 1')
                if (messageBox.innerHTML == '...') {
                    console.log('ani 2')
                    messageBox.innerHTML = '.'
                } else {
                    console.log('ani 3')
                    messageBox.innerHTML += '.';
                }
                console.log('ani 4')
                await this.sleep(1)
            }
        }

    },
    async mounted () {

        try {

            var python2 = new RegExp('.*print(.*).*');
            var code = "print('Hello, World!')";
            if (code.match(python2)) {
                console.log("The string contains a print statement.");
            } else {
                console.log("The string does not contain a print statement.");
            }

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

    }
    
})

app.mount('#vue-app')