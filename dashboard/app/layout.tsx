import './globals.css';
import type {Metadata} from 'next';
export const metadata: Metadata = {title:'Undercover / Episode Desk',description:'Local forensic viewer for research episode ledgers'};
export default function Layout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
