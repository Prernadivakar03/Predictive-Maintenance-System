import { useEffect, useState } from "react";

// Hash routing keeps the app deployable behind FastAPI's static files without server-side route fallbacks.
const read = () => (window.location.hash.replace(/^#/, "") || "/").split("?")[0];

export function useRoute() {
  const [path, setPath] = useState(read);
  useEffect(() => {
    const onChange = () => { setPath(read()); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return path;
}

export const href = (path) => `#${path}`;
export const navigate = (path) => { window.location.hash = path; };
