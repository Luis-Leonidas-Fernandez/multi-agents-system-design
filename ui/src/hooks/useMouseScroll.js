import React from 'react';
import {mouseEvents} from '../utils/mouseFilter.js';

/**
 * Suscribe al mouse wheel event emitido por el Transform filter de mouseFilter.js.
 * El filtrado y la habilitación del mouse tracking se hacen en cli.js,
 * antes de que Ink tome el control del stdin.
 */
export function useMouseScroll(onWheel) {
  const cbRef = React.useRef(onWheel);
  React.useLayoutEffect(() => { cbRef.current = onWheel; }, [onWheel]);

  React.useEffect(() => {
    const handler = (dir) => cbRef.current?.(dir);
    mouseEvents.on('wheel', handler);
    return () => mouseEvents.off('wheel', handler);
  }, []);
}
