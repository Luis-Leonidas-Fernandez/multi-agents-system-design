import React from 'react';
import {render} from 'ink';
import {App} from '../components/App.js';
import {REPL} from '../screens/REPL.js';
import {createMouseFilter} from '../utils/mouseFilter.js';

const MOUSE_ENABLE  = '\x1b[?1000h\x1b[?1006h';
const MOUSE_DISABLE = '\x1b[?1000l\x1b[?1006l';

function disableMouse() {
  try { process.stdout.write(MOUSE_DISABLE); } catch {}
}

function main() {
  const isTTY = process.stdin.isTTY === true && process.stdout.isTTY === true;

  // Crear el stream filtrado ANTES de que Ink toque stdin.
  // Ink leerá del stream filtrado → nunca ve escape codes de mouse.
  const stdin = isTTY ? createMouseFilter(process.stdin) : process.stdin;

  if (isTTY) {
    process.stdout.write(MOUSE_ENABLE);
    process.once('exit',   disableMouse);
    process.once('SIGINT',  () => { disableMouse(); process.exit(0); });
    process.once('SIGTERM', () => { disableMouse(); process.exit(0); });
  }

  render(
    React.createElement(App, null, React.createElement(REPL)),
    {
      stdin,
      isRawModeSupported: isTTY,
      exitOnCtrlC: true,
    },
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export {main};
