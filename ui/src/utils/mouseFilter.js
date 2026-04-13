/**
 * Transform stream que filtra eventos de mouse ANSI antes de que lleguen a Ink.
 *
 * Ink 6 lee stdin via su propio `stdin.on('data', ...)`. Si pasamos el stream
 * crudo, los escape codes de mouse (\x1b[<Pb;Px;PyM) llegan a la pipeline de
 * Ink, que no los reconoce y los trata como texto.
 *
 * Solución: interceptar al nivel del stream, no al nivel de listeners de React.
 */
import {Transform} from 'stream';
import {EventEmitter} from 'events';

// Bus interno — useMouseScroll se suscribe acá
export const mouseEvents = new EventEmitter();

// SGR mouse: ESC [ < Pb ; Px ; Py M/m  (buttons, clicks, wheel)
const RE_SGR_ALL    = /\x1b\[<\d+;\d+;\d+[Mm]/g;
const RE_SGR_WHEEL  = /\x1b\[<(\d+);\d+;\d+[Mm]/g;

/**
 * Crea un Transform que:
 *  1. Detecta wheel events (button 64/65) y los emite en mouseEvents
 *  2. Elimina TODAS las secuencias SGR de mouse del buffer
 *  3. Proxea isTTY y setRawMode al source real
 */
export function createMouseFilter(source) {
  const filter = new Transform({
    transform(chunk, _enc, cb) {
      const raw = chunk.toString('binary');

      // Emitir wheel events
      RE_SGR_WHEEL.lastIndex = 0;
      let m;
      while ((m = RE_SGR_WHEEL.exec(raw)) !== null) {
        const btn = parseInt(m[1], 10);
        if (btn === 64) mouseEvents.emit('wheel', 'up');
        else if (btn === 65) mouseEvents.emit('wheel', 'down');
      }

      // Eliminar todas las secuencias SGR de mouse
      const filtered = raw.replace(RE_SGR_ALL, '');
      if (filtered.length > 0) cb(null, Buffer.from(filtered, 'binary'));
      else cb();
    },
  });

  // Copiar propiedades TTY para que Ink no se queje
  filter.isTTY = source.isTTY;
  Object.defineProperty(filter, 'rows',    { get: () => source.rows });
  Object.defineProperty(filter, 'columns', { get: () => source.columns });

  // Ink llama setRawMode en el stdin que recibe — lo redirigimos al TTY real
  filter.setRawMode = (mode) => { source.setRawMode?.(mode); return filter; };

  // net.Socket methods que Ink 6 llama para controlar el event loop
  filter.ref   = () => { source.ref?.();   return filter; };
  filter.unref = () => { source.unref?.(); return filter; };

  // Flujo de datos: source → filter (sin cerrar filter al cerrar source)
  source.on('data', (chunk) => filter.write(chunk));
  source.on('end',  ()      => filter.end());

  return filter;
}
