import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire('/Users/eladkulman/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/anchor.cjs');
const {PresentationFile,FileBlob}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')));
const p=await PresentationFile.importPptx(await FileBlob.load('final report/qqq_chronos2_presentation_v4.pptx'));
const out='tmp/lora_slide/final';await fs.mkdir(out,{recursive:true});
const results=[];
for(let i=0;i<p.slides.items.length;i++){
 const name=`slide-${String(i+1).padStart(2,'0')}.png`;
 const png=new Uint8Array(await (await p.slides.items[i].export({format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(`${out}/${name}`,png);
 if(i<16){
 const old=await fs.readFile(`tmp/final_report_deck/final/${name}`);
 const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
 results.push({slide:i+1,identical:hash(old)===hash(png)});
 }
}
await fs.writeFile(`${out}/comparison.json`,JSON.stringify(results,null,2));
console.log(JSON.stringify({slides:p.slides.items.length,unchanged:results}));
