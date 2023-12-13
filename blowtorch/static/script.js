const { createApp, ref } = Vue

createApp({
    data () {
        return {
            message: 'Hello Vue!',
            context: {},
            count: 0
        }
    },
    methods: {
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
        submit () {

            const inputField = document.getElementById('msg-input-field');
            const payload = inputField.value;
            inputField.value = '';
            
            // build message box for chat window
            this.messageBox(payload, 'r')
        }

    },
    mounted () {
        try {
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