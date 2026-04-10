import React from 'react';
import {render} from 'ink';
import {App} from '../components/App.js';
import {REPL} from '../screens/REPL.js';

function main() {
  render(React.createElement(App, null, React.createElement(REPL)), {
    isRawModeSupported: process.stdin.isTTY === true,
    exitOnCtrlC: true,
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export {main};
