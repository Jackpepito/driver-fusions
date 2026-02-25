"""
Core fusion reconstruction logic for building isoform-aware fusion proteins.

This module provides the main reconstruction pipeline that:
- Parses GTF files to extract transcript structures
- Builds CDS sequences from genomic coordinates
- Maps fusion breakpoints to CDS space
- Reconstructs fusion proteins from all isoform combinations
- Uses ORFfinder for optimal ORF detection
"""

import re
import requests
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq


# ============================================================================
# GENETIC CODE
# ============================================================================

STANDARD_GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# ============================================================================
# TRANSLATION FUNCTIONS
# ============================================================================

def translate_seq(cds: str, stop_symbol: str = "*") -> str:
    """
    Traduce una sequenza CDS in proteina usando STANDARD_GENETIC_CODE.
    Si ferma al primo stop codon. Se non multiplo di 3, taglia residuo.
    """
    protein = []
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i:i + 3].upper()
        aa = STANDARD_GENETIC_CODE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def run_orffinder(
    dna_seq: str,
    orffinder_path: str = "/homes/gcapitani/Gene-Fusions/data/ORFfinder",
    min_len: int = 30,
    timeout: int = 60,
    verbose: bool = False
) -> Tuple[str, int, int, int]:
    """
    Runs NCBI ORFfinder on a DNA sequence to find the longest/best ORF.
    
    Args:
        dna_seq: DNA sequence to analyze
        orffinder_path: Path to ORFfinder executable
        min_len: Minimum ORF length in nucleotides (default 30 = 10 aa)
        timeout: Maximum execution time in seconds
        verbose: Print debug information
    
    Returns:
        Tuple of (protein_sequence, orf_start_in_dna, orf_end_in_dna, frame)
        Returns ("", -1, -1, -1) if no ORF found or error
    
    Notes:
        - ORFfinder output format (-outfmt 0): ">lcl|ORF1_seqname:start:end [gcode=1] [location=123..456]"
        - Start/end in header are 0-based in DNA coordinates
        - Frame is calculated from start position modulo 3
    """
    if not dna_seq or len(dna_seq) < min_len:
        return "", -1, -1, -1
    
    # Create temporary input/output files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f_in:
        f_in.write(f">seq\n{dna_seq}\n")
        input_file = f_in.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f_out:
        output_file = f_out.name
    
    try:
        # Run ORFfinder with protein FASTA output
        cmd = [
            orffinder_path,
            "-in", input_file,
            "-out", output_file,
            "-outfmt", "0",  # Protein FASTA format
            "-ml", str(min_len),
            "-n", "true",  # Find ORFs on both strands
            "-s", "0"  # Start codon: 0 = ATG only
        ]
        
        if verbose:
            print(f"[ORFfinder] Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"[ORFfinder] Error (returncode={result.returncode}): {result.stderr}")
            return "", -1, -1, -1
        
        # Parse output - take first (longest) ORF
        if not Path(output_file).exists():
            if verbose:
                print("[ORFfinder] No output file generated")
            return "", -1, -1, -1
        
        with open(output_file) as f:
            lines = f.readlines()
            if len(lines) < 2:
                if verbose:
                    print("[ORFfinder] No ORFs found")
                return "", -1, -1, -1
            
            # Parse header: ">lcl|ORF1_seq:123:456"
            header = lines[0].strip()
            protein = ''.join(line.strip() for line in lines[1:] if not line.startswith('>'))
            
            # Extract coordinates from header
            # Format: >lcl|ORFn_seqname:start:end
            coord_match = re.search(r':(\d+):(\d+)', header)
            if not coord_match:
                if verbose:
                    print(f"[ORFfinder] Could not parse coordinates from: {header}")
                # Return protein without coordinates
                return protein, -1, -1, -1
            
            orf_start = int(coord_match.group(1))
            orf_end = int(coord_match.group(2))
            
            # Calculate frame (0, 1, or 2) from start position
            frame = orf_start % 3
            
            if verbose:
                print(f"[ORFfinder] Found ORF: {orf_start}:{orf_end} (frame {frame}), length={len(protein)} aa")
            
            return protein, orf_start, orf_end, frame
        
    except subprocess.TimeoutExpired:
        if verbose:
            print("[ORFfinder] Timeout expired")
        return "", -1, -1, -1
    except Exception as e:
        if verbose:
            print(f"[ORFfinder] Error: {e}")
        return "", -1, -1, -1
    finally:
        # Cleanup temp files
        try:
            Path(input_file).unlink()
            Path(output_file).unlink()
        except:
            pass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Exon:
    start: int
    end: int
    strand: int


@dataclass
class Transcript:
    transcript_id: str
    chromosome: str
    strand: int
    exons: List[Exon]
    cds_start: Optional[int]  # genomic coordinate
    cds_end: Optional[int]    # genomic coordinate


@dataclass
class FusionIsoformResult:
    gene5: str
    gene3: str
    tx5_id: str
    tx3_id: str
    bp5: int
    bp3: int
    bp5_in_cds: bool
    bp3_in_cds: bool
    bp5_approximated: bool  # True se breakpoint approssimato
    bp3_approximated: bool  # True se breakpoint approssimato
    in_frame: bool
    cds5_len: int
    cds3_len: int
    fusion_cds_len: int
    protein_seq: str
    quality: str = "perfect"  # 'perfect', 'good', 'approximate', 'out_of_frame'
    note: str = ""
    # ORFfinder results
    orf_used: bool = False  # True se è stato usato ORFfinder
    orf_start: int = -1  # Posizione start ORF nella fusion_cds
    orf_end: int = -1  # Posizione end ORF
    orf_frame: int = -1  # Frame dell'ORF (0, 1, 2)


# ============================================================================
# MAIN RECONSTRUCTION PIPELINE
# ============================================================================

class IsoformAwareFusionReconstructor:
    """
    Pipeline che usa:
    - GTF locale per recupero trascritti e coordinate CDS [mode='local']
    - Genoma locale (.2bit) per sequenze genomiche [mode='local']
    - Oppure API Ensembl/UCSC [mode='api']
    - Ricostruzione CDS e traduzione in proteina per tutte le isoforme
    """

    def __init__(self, mode: str = 'local', gtf_path: str = None, genome_path: str = None, 
                 genome_build: str = "hg38", cache_dir: str = "./cache",
                 use_orffinder: bool = True,
                 orffinder_path: str = "/homes/gcapitani/Gene-Fusions/data/ORFfinder"):
        """
        Args:
            mode: 'local' (usa GTF+genome locali) o 'api' (usa Ensembl/UCSC APIs)
            gtf_path: Path al file GTF (solo per mode='local')
            genome_path: Path al genoma FASTA (.fa/.fasta) (solo per mode='local')
            genome_build: "hg19" o "hg38" (solo per mode='api')
            cache_dir: Directory per cache dei trascritti parsati
            use_orffinder: Se True, usa ORFfinder per trovare l'ORF migliore
            orffinder_path: Path all'eseguibile ORFfinder
        """
        self.mode = mode
        self.genome_build = genome_build
        self.cache_dir = Path(cache_dir)
        # Ensure nested cache paths exist regardless of current working directory.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_orffinder = use_orffinder
        self.orffinder_path = orffinder_path
        
        print(f"[Init] Mode: {mode}")
        if use_orffinder:
            print(f"[Init] ORFfinder: enabled ({orffinder_path})")
        else:
            print(f"[Init] ORFfinder: disabled")
        
        if mode == 'local':
            if not gtf_path or not genome_path:
                raise ValueError("mode='local' requires gtf_path and genome_path")
            
            self.gtf_path = Path(gtf_path)
            self.genome_path = Path(genome_path)
            
            print(f"[Init] GTF file: {self.gtf_path}")
            print(f"[Init] Genome file: {self.genome_path}")
            print(f"[Init] Cache dir: {self.cache_dir}")
            
            # Load genome into memory (dict: chr -> Seq)
            print("[Init] Loading genome FASTA (this may take a few minutes)...")
            self.genome = self._load_genome_fasta(str(self.genome_path))
            print(f"[Init] Loaded {len(self.genome)} chromosomes")
            
            # Cache for transcripts (gene_symbol -> List[Transcript])
            self.transcript_cache: Dict[str, List[Transcript]] = {}
            
            # Try to load cached GTF parse
            self._load_gtf_cache()
            
        elif mode == 'api':
            self.ensembl_rest = (
                "https://grch37.rest.ensembl.org"
                if genome_build == "hg19"
                else "https://rest.ensembl.org"
            )
            self.ucsc_api = "https://api.genome.ucsc.edu"
            print(f"[Init] Ensembl REST: {self.ensembl_rest}")
            print(f"[Init] UCSC API: {self.ucsc_api}")
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'local' or 'api'")
    
    # ========== LOCAL MODE METHODS ==========
    
    def _load_genome_fasta(self, fasta_path: str) -> Dict[str, Seq]:
        """Load genome FASTA into memory."""
        genome = {}
        with open(fasta_path) as f:
            for record in SeqIO.parse(f, "fasta"):
                genome[record.id] = record.seq
        return genome
    
    def _load_gtf_cache(self):
        """Load pre-parsed GTF from cache if available."""
        import pickle
        cache_file = self.cache_dir / "gtf_transcripts.pkl"
        
        if cache_file.exists():
            print(f"[Cache] Loading GTF from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    self.transcript_cache = pickle.load(f)
                print(f"[Cache] Loaded {len(self.transcript_cache)} genes from cache")
                return
            except Exception as e:
                print(f"[Cache] Failed to load cache: {e}")
        
        print("[GTF] Parsing GTF file (first time, will be cached)...")
        self._parse_gtf()
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.transcript_cache, f)
            print(f"[Cache] Saved {len(self.transcript_cache)} genes to cache")
        except Exception as e:
            print(f"[Cache] Failed to save cache: {e}")
    
    def _parse_gtf(self):
        """Parse GTF file and build transcript structures."""
        import gzip
        
        # Check if gzipped
        open_func = gzip.open if str(self.gtf_path).endswith('.gz') else open
        
        # Temp storage: transcript_id -> data
        tx_data: Dict[str, dict] = {}
        gene_to_symbol: Dict[str, str] = {}  # gene_id -> gene_name
        
        print("[GTF] Reading file...")
        with open_func(self.gtf_path, 'rt') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                chrom, source, feature, start, end, score, strand, frame, info = parts
                start, end = int(start), int(end)
                strand_int = 1 if strand == '+' else -1
                
                # Parse info field
                gene_id_match = re.search(r'gene_id "([^"]+)"', info)
                gene_name_match = re.search(r'gene_name "([^"]+)"', info)
                tx_id_match = re.search(r'transcript_id "([^"]+)"', info)
                
                if not tx_id_match:
                    continue
                
                tx_id = tx_id_match.group(1)
                gene_id = gene_id_match.group(1) if gene_id_match else None
                gene_name = gene_name_match.group(1) if gene_name_match else None
                
                if gene_id and gene_name:
                    gene_to_symbol[gene_id] = gene_name
                
                # Initialize transcript entry
                if tx_id not in tx_data:
                    tx_data[tx_id] = {
                        'transcript_id': tx_id,
                        'gene_name': gene_name,
                        'chromosome': chrom.replace('chr', ''),
                        'strand': strand_int,
                        'exons': [],
                        'cds_regions': []
                    }
                
                # Collect features
                if feature == 'exon':
                    tx_data[tx_id]['exons'].append((start, end))
                elif feature == 'CDS':
                    tx_data[tx_id]['cds_regions'].append((start, end))
                
                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines...")
        
        print(f"[GTF] Found {len(tx_data)} transcripts")
        
        # Convert to Transcript objects and group by gene
        for tx_id, data in tx_data.items():
            gene_name = data['gene_name']
            if not gene_name:
                continue
            
            # Sort exons
            exons_sorted = sorted(data['exons'], key=lambda x: x[0], 
                                reverse=(data['strand'] == -1))
            
            # Determine CDS start/end (min/max of all CDS regions)
            cds_start, cds_end = None, None
            if data['cds_regions']:
                cds_start = min(r[0] for r in data['cds_regions'])
                cds_end = max(r[1] for r in data['cds_regions'])
            
            # Create Exon objects
            exon_objs = [Exon(start=s, end=e, strand=data['strand']) 
                        for s, e in exons_sorted]
            
            transcript = Transcript(
                transcript_id=tx_id,
                chromosome=data['chromosome'],
                strand=data['strand'],
                exons=exon_objs,
                cds_start=cds_start,
                cds_end=cds_end
            )
            
            if gene_name not in self.transcript_cache:
                self.transcript_cache[gene_name] = []
            self.transcript_cache[gene_name].append(transcript)
        
        print(f"[GTF] Organized into {len(self.transcript_cache)} genes")
    
    def _get_transcripts_local(self, gene_symbol: str) -> List[Transcript]:
        """Get transcripts from local GTF cache."""
        if gene_symbol not in self.transcript_cache:
            print(f"[GTF] WARNING: Gene '{gene_symbol}' not found in GTF")
            return []
        
        transcripts = self.transcript_cache[gene_symbol]
        print(f"[GTF] Found {len(transcripts)} transcripts for {gene_symbol}")
        return transcripts
    
    def _get_sequence_local(self, chromosome: str, start: int, end: int, strand: int) -> str:
        """Get genomic sequence from local FASTA file."""
        # Add chr prefix if needed
        if not chromosome.startswith("chr"):
            chrom_key = f"chr{chromosome}"
        else:
            chrom_key = chromosome
        
        if chrom_key not in self.genome:
            print(f"[Genome] WARNING: Chromosome '{chrom_key}' not found")
            return ""
        
        # Extract sequence (1-based coordinates)
        try:
            seq = str(self.genome[chrom_key][start-1:end]).upper()
        except Exception as e:
            print(f"[Genome] ERROR extracting {chrom_key}:{start}-{end}: {e}")
            return ""
        
        # Reverse complement if negative strand
        if strand == -1:
            seq = str(Seq(seq).reverse_complement())
        
        return seq
    
    # ========== API MODE METHODS ==========
    
    def _get(self, url: str, params: Dict = None) -> Optional[dict]:
        """Helper for API requests."""
        headers = {"Content-Type": "application/json"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            if r.text.strip():
                return r.json()
            return None
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] GET {url}: {e}")
            return None
    
    def _get_transcripts_api(self, gene_symbol: str) -> List[Transcript]:
        """Get transcripts from Ensembl API."""
        print(f"\n[Ensembl] Querying gene: {gene_symbol}")
        lookup_url = f"{self.ensembl_rest}/lookup/symbol/homo_sapiens/{gene_symbol}"
        params = {"expand": "1"}
        data = self._get(lookup_url, params=params)
        if data is None:
            print("[Ensembl] No data for gene.")
            return []

        chrom = data.get("seq_region_name")
        strand = data.get("strand")
        txs: List[Transcript] = []

        transcript_entries = data.get("Transcript", [])
        print(f"[Ensembl] Found {len(transcript_entries)} transcripts for {gene_symbol}")

        for t in transcript_entries:
            exons: List[Exon] = []
            for ex in t.get("Exon", []):
                exons.append(
                    Exon(
                        start=ex["start"],
                        end=ex["end"],
                        strand=ex["strand"],
                    )
                )

            cds_start = None
            cds_end = None
            if "Translation" in t:
                cds_start = t["Translation"]["start"]
                cds_end = t["Translation"]["end"]

            tx = Transcript(
                transcript_id=t["id"],
                chromosome=chrom,
                strand=strand,
                exons=sorted(
                    exons,
                    key=lambda e: e.start,
                    reverse=(strand == -1),
                ),
                cds_start=cds_start,
                cds_end=cds_end,
            )
            txs.append(tx)

        return txs
    
    def _get_sequence_api(self, chromosome: str, start: int, end: int, strand: int) -> str:
        """Get genomic sequence from UCSC API."""
        # Convert 1-based to 0-based for UCSC
        start = start - 1
        
        if not chromosome.startswith("chr"):
            chromosome = f"chr{chromosome}"

        url = f"{self.ucsc_api}/getData/sequence"
        params = {"genome": self.genome_build, "chrom": chromosome, "start": start, "end": end}
        data = self._get(url, params=params)
        if data is None or "dna" not in data:
            print(f"[UCSC] WARNING: no data for {chromosome}:{start}-{end}")
            return ""
        dna = data["dna"].upper()
        if strand == -1:
            dna = str(Seq(dna).reverse_complement())
        return dna
    
    # ========== UNIFIED PUBLIC METHODS ==========

    def get_gene_transcripts(self, gene_symbol: str) -> List[Transcript]:
        """Get transcripts (mode-agnostic)."""
        if self.mode == 'local':
            return self._get_transcripts_local(gene_symbol)
        else:
            return self._get_transcripts_api(gene_symbol)

    def get_genomic_sequence(self, chromosome: str, start: int, end: int, strand: int) -> str:
        """Get genomic sequence (mode-agnostic). Coordinates: 1-based inclusive."""
        if self.mode == 'local':
            return self._get_sequence_local(chromosome, start, end, strand)
        else:
            return self._get_sequence_api(chromosome, start, end, strand)

    # ---------- Costruzione CDS di un trascritto ----------

    def build_cds_sequence(self, tx: Transcript) -> Tuple[str, List[Tuple[int, int, int, int]]]:
        """
        Costruisce la CDS di un trascritto concatenando i pezzi degli esoni
        che ricadono nella regione [cds_start, cds_end] genomica.
        Ritorna:
          - cds_seq (5'->3' nella direzione del trascritto)
          - lista di mapping (g_start, g_end, cds_start, cds_end) in coordinate 0-based CDS.
        """
        if tx.cds_start is None or tx.cds_end is None:
            return "", []

        cds_gstart, cds_gend = tx.cds_start, tx.cds_end

        cds_seq = []
        mapping: List[Tuple[int, int, int, int]] = []
        cds_pos = 0  # indice nella CDS concatenata

        # Attenzione: gli esoni sono già ordinati secondo lo strand
        for ex in tx.exons:
            # Intersezione esone/CDS a livello genomico
            ex_start = ex.start
            ex_end = ex.end

            # Non usare regioni fuori dalla CDS
            seg_start = max(ex_start, cds_gstart)
            seg_end = min(ex_end, cds_gend)
            if seg_start > seg_end:
                continue

            seg_seq = self.get_genomic_sequence(
                tx.chromosome, seg_start, seg_end, tx.strand
            )
            seg_len = len(seg_seq)
            if seg_len == 0:
                continue

            cds_seq.append(seg_seq)
            mapping.append((seg_start, seg_end, cds_pos, cds_pos + seg_len))
            cds_pos += seg_len

        full_cds = "".join(cds_seq)
        return full_cds, mapping

    # ---------- Mappa breakpoint genomico alla CDS ----------

    @staticmethod
    def map_breakpoint_to_cds(
        breakpoint: int, mapping: List[Tuple[int, int, int, int]]
    ) -> Optional[int]:
        """
        Dato un breakpoint in coordinate genomiche, ritorna la posizione
        corrispondente nella CDS (0-based), usando il mapping (g_start,g_end,cds_start,cds_end).
        Se il breakpoint cade fuori da tutte le regioni CDS, ritorna None.
        """
        for g_start, g_end, cds_start, cds_end in mapping:
            if g_start <= breakpoint <= g_end:
                # Offset dal g_start
                offset = breakpoint - g_start
                return cds_start + offset
        return None

    # ---------- Estrazione frammento CDS head / tail ----------

    def extract_cds_fragment(
        self,
        tx: Transcript,
        breakpoint: int,
        role: str,
        allow_approximation: bool = True,
    ) -> Tuple[str, bool, bool, str]:
        """
        Estrae un frammento CDS "head" (5') o "tail" (3') rispetto a breakpoint,
        lavorando in spazio CDS.

        Args:
            tx: Transcript object
            breakpoint: genomic coordinate
            role: 'head' or 'tail'
            allow_approximation: if True, approximate breakpoint to nearest CDS boundary

        Ritorna:
          - frammento CDS (string)
          - flag is_in_cds (bool) - True se breakpoint esatto in CDS
          - flag is_approximated (bool) - True se breakpoint approssimato
          - nota descrittiva (str)
        """
        cds_seq, mapping = self.build_cds_sequence(tx)
        if not cds_seq or not mapping:
            return "", False, False, "Transcript non codificante (no CDS)."

        cds_pos = self.map_breakpoint_to_cds(breakpoint, mapping)
        is_approximated = False
        
        if cds_pos is None:
            # Breakpoint fuori dalla CDS
            if not allow_approximation:
                return "", False, False, "Breakpoint fuori dalla CDS per questo trascritto."
            
            # APPROSSIMAZIONE: usa il boundary CDS più vicino
            if tx.cds_start is None or tx.cds_end is None:
                return "", False, False, "No CDS boundaries available."
            
            # Determina se il breakpoint è prima o dopo la CDS
            if breakpoint < tx.cds_start:
                # Prima della CDS → usa inizio CDS
                cds_pos = 0
                is_approximated = True
            elif breakpoint > tx.cds_end:
                # Dopo la CDS → usa fine CDS
                cds_pos = len(cds_seq)
                is_approximated = True
            else:
                # Dentro un introne tra esoni CDS → usa boundary più vicino
                # Calcola distanze ai boundaries
                dist_start = abs(breakpoint - tx.cds_start)
                dist_end = abs(breakpoint - tx.cds_end)
                if dist_start < dist_end:
                    cds_pos = 0
                else:
                    cds_pos = len(cds_seq)
                is_approximated = True

        # Taglio in spazio CDS
        if role == "head":
            frag = cds_seq[:cds_pos]
            note = f"Head: CDS[0:{cds_pos}]" + (" (approx)" if is_approximated else "")
        elif role == "tail":
            frag = cds_seq[cds_pos:]
            note = f"Tail: CDS[{cds_pos}:end]" + (" (approx)" if is_approximated else "")
        else:
            raise ValueError("role deve essere 'head' o 'tail'.")

        return frag, not is_approximated, is_approximated, note

    # ---------- Ricostruzione di tutte le fusioni isoforma-aware ----------

    def reconstruct_isoform_fusions(
        self,
        gene5: str,
        chr5: str,
        bp5: int,
        gene3: str,
        chr3: str,
        bp3: int,
        min_cds_len: int = 30,  # ABBASSATO da 75 a 30
        allow_approximation: bool = True,
        allow_out_of_frame: bool = True,
    ) -> List[FusionIsoformResult]:
        """
        Ricostruisce tutte le fusioni codificanti possibili fra isoforme di gene5 (partner 5')
        e isoforme di gene3 (partner 3') dato un paio di breakpoint genomici.

        Args:
            min_cds_len: lunghezza minima CDS fragment (default 30nt = 10aa)
            allow_approximation: se True, approssima breakpoint fuori CDS
            allow_out_of_frame: se True, accetta fusioni out-of-frame

        Ritorna una lista di FusionIsoformResult.
        """
        print("\n" + "=" * 80)
        print(f"[Fusion] {gene5} (chr{chr5}:{bp5}) -- {gene3} (chr{chr3}:{bp3})")
        print("=" * 80)

        txs5 = self.get_gene_transcripts(gene5)
        txs3 = self.get_gene_transcripts(gene3)

        if not txs5 or not txs3:
            print("[Fusion] Nessun trascritto trovato per uno dei geni.")
            return []

        # Filtra per cromosoma coerente (opzionale ma sensato)
        # Normalizza cromosomi: rimuovi 'chr' prefix per confronto robusto
        def normalize_chr(c):
            c_str = str(c).strip()
            if c_str.lower().startswith('chr'):
                return c_str[3:]
            return c_str
        
        chr5_norm = normalize_chr(chr5) if chr5 else None
        chr3_norm = normalize_chr(chr3) if chr3 else None
        
        if chr5_norm:
            txs5 = [t for t in txs5 if normalize_chr(t.chromosome) == chr5_norm]
            if not txs5:
                print(f"[Warning] No transcripts found for {gene5} on chr{chr5_norm}")
                return []
        
        if chr3_norm:
            txs3 = [t for t in txs3 if normalize_chr(t.chromosome) == chr3_norm]
            if not txs3:
                print(f"[Warning] No transcripts found for {gene3} on chr{chr3_norm}")
                return []

        results: List[FusionIsoformResult] = []

        for tx5 in txs5:
            for tx3 in txs3:
                # Estrai frammenti CDS head/tail con approssimazione
                cds_head, in_cds5, approx5, note5 = self.extract_cds_fragment(
                    tx5, bp5, role="head", allow_approximation=allow_approximation
                )
                cds_tail, in_cds3, approx3, note3 = self.extract_cds_fragment(
                    tx3, bp3, role="tail", allow_approximation=allow_approximation
                )

                # Se non abbiamo sequence, skip
                if not cds_head or not cds_tail:
                    continue

                # Controlla lunghezza minima
                if len(cds_head) < min_cds_len or len(cds_tail) < min_cds_len:
                    continue

                fusion_cds = cds_head + cds_tail

                # Controllo frame
                in_frame = (len(fusion_cds) % 3 == 0)
                
                # Se out-of-frame e non accettiamo, skip
                if not in_frame and not allow_out_of_frame:
                    continue

                # TRADUZIONE CON ORFFINDER
                orf_used = False
                orf_start, orf_end, orf_frame = -1, -1, -1
                
                if self.use_orffinder:
                    # Usa ORFfinder per trovare l'ORF migliore
                    protein, orf_start, orf_end, orf_frame = run_orffinder(
                        fusion_cds, 
                        orffinder_path=self.orffinder_path,
                        min_len=min_cds_len
                    )
                    
                    if protein:  # ORFfinder ha trovato un ORF
                        orf_used = True
                        # Aggiorna in_frame based on ORF frame
                        # Se l'ORF è in frame 0, la fusione è considerata in-frame
                        if orf_frame == 0:
                            in_frame = True
                    else:
                        # Fallback: traduzione diretta se ORFfinder fallisce
                        protein = translate_seq(fusion_cds)
                else:
                    # Traduzione diretta senza ORFfinder
                    protein = translate_seq(fusion_cds)

                # Determina quality score
                if in_cds5 and in_cds3 and in_frame:
                    quality = "perfect"
                elif in_cds5 and in_cds3 and not in_frame:
                    quality = "out_of_frame"
                elif (approx5 or approx3) and in_frame:
                    quality = "approximate"
                elif (approx5 or approx3) and not in_frame:
                    quality = "approximate_out_of_frame"
                else:
                    quality = "good"

                note = f"{note5} | {note3}"
                if orf_used:
                    note += f" | ORF[{orf_start}:{orf_end}] frame={orf_frame}"
                
                result = FusionIsoformResult(
                    gene5=gene5,
                    gene3=gene3,
                    tx5_id=tx5.transcript_id,
                    tx3_id=tx3.transcript_id,
                    bp5=bp5,
                    bp3=bp3,
                    bp5_in_cds=in_cds5,
                    bp3_in_cds=in_cds3,
                    bp5_approximated=approx5,
                    bp3_approximated=approx3,
                    in_frame=in_frame,
                    cds5_len=len(cds_head),
                    cds3_len=len(cds_tail),
                    fusion_cds_len=len(fusion_cds),
                    protein_seq=protein,
                    quality=quality,
                    note=note,
                    orf_used=orf_used,
                    orf_start=orf_start,
                    orf_end=orf_end,
                    orf_frame=orf_frame,
                )
                results.append(result)

        print(f"[Fusion] Trovate {len(results)} fusioni isoforma-specifiche codificanti.")
        return results


# ============================================================================
# LOGGING AND ANALYSIS UTILITIES
# ============================================================================

def log_fusion_result(idx, total, gene5, gene3, chr5, chr3, bp5, bp3, 
                      original_seq, fusion_results, dataset_name=""):
    """
    Standard logging function for fusion reconstruction results.
    
    Args:
        idx: Current fusion index (0-based)
        total: Total number of fusions
        gene5: 5' gene name
        gene3: 3' gene name
        chr5: 5' chromosome
        chr3: 3' chromosome
        bp5: 5' breakpoint
        bp3: 3' breakpoint
        original_seq: Original sequence (if available)
        fusion_results: List of FusionResult objects
        dataset_name: Name of dataset for logging
    """
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] {gene5}-{gene3}")
    if dataset_name:
        print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"  Chromosomes: chr{chr5}:{bp5} - chr{chr3}:{bp3}")
    
    # Original sequence info
    if original_seq and original_seq != "":
        print(f"  Original sequence length: {len(original_seq)} aa")
    else:
        print(f"  Original sequence: Not available")
    
    # Reconstruction results
    print(f"  Isoform combinations found: {len(fusion_results)}")
    
    if fusion_results:
        best = fusion_results[0]
        print(f"  Best result:")
        print(f"    - Reconstructed length: {len(best.protein_seq)} aa")
        print(f"    - Transcripts: {best.tx5_id} + {best.tx3_id}")
        print(f"    - Quality: {best.quality}")
        print(f"    - In-frame: {best.in_frame}")
        print(f"    - ORF used: {best.orf_used}")


def generate_final_analysis(results_df, output_prefix, dataset_name="Dataset"):
    """
    Generate comprehensive final analysis including plots and amino acid distribution.
    All outputs are saved in a dedicated folder named after the output_prefix.
    
    Args:
        results_df: DataFrame with reconstruction results
        output_prefix: Prefix for output files (will create folder with this name)
        dataset_name: Name of dataset for titles
    """
    import pandas as pd
    import numpy as np
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    # Create output directory (use absolute path)
    output_dir = Path(f"{output_prefix}_results").absolute()
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"FINAL ANALYSIS - {dataset_name}")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}/")
    print()
    
    # Standard amino acids
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print(f"\nSummary Statistics:")
    print(f"  Total fusions: {len(results_df)}")
    print(f"  Mean reconstructed length: {results_df['reconstructed_length'].mean():.1f} aa")
    print(f"  Median reconstructed length: {results_df['reconstructed_length'].median():.1f} aa")
    
    if 'original_length' in results_df.columns:
        has_original = (results_df['original_length'] > 0).sum()
        if has_original > 0:
            print(f"  Fusions with original sequence: {has_original}")
            print(f"  Mean original length: {results_df[results_df['original_length']>0]['original_length'].mean():.1f} aa")
    
    if 'identity' in results_df.columns:
        valid_identity = results_df[results_df['identity'] > 0]
        if len(valid_identity) > 0:
            print(f"  Mean identity: {valid_identity['identity'].mean():.2%}")
            print(f"  Median identity: {valid_identity['identity'].median():.2%}")
    
    if 'n_isoforms' in results_df.columns:
        print(f"  Mean isoforms per fusion: {results_df['n_isoforms'].mean():.1f}")
        print(f"  Median isoforms: {results_df['n_isoforms'].median():.0f}")
    
    if 'in_frame' in results_df.columns:
        in_frame_count = results_df['in_frame'].sum()
        print(f"  In-frame fusions: {in_frame_count} ({100*in_frame_count/len(results_df):.1f}%)")
    
    if 'quality' in results_df.columns:
        print(f"\n  Quality distribution:")
        for quality, count in results_df['quality'].value_counts().items():
            print(f"    {quality}: {count} ({100*count/len(results_df):.1f}%)")
    
    # ========================================================================
    # AMINO ACID DISTRIBUTION
    # ========================================================================
    print(f"\n{'='*80}")
    print("AMINO ACID DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze reconstructed sequences
    reconstructed_seqs = results_df['reconstructed_seq'].dropna().tolist()
    if reconstructed_seqs:
        all_aa_recon = ''.join(reconstructed_seqs)
        aa_counts_recon = Counter(all_aa_recon)
        total_recon = sum(aa_counts_recon.values())
        aa_freq_recon = {aa: aa_counts_recon.get(aa, 0) / total_recon * 100 for aa in AMINO_ACIDS}
        
        print(f"\nReconstructed sequences:")
        print(f"  Total residues: {total_recon:,}")
        sorted_aa = sorted(aa_freq_recon.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 5 amino acids: {', '.join([f'{aa}={freq:.2f}%' for aa, freq in sorted_aa[:5]])}")
    
    # Analyze original sequences if available
    aa_freq_orig = None
    if 'original_seq' in results_df.columns:
        original_seqs = results_df['original_seq'].dropna()
        original_seqs = original_seqs[original_seqs != ''].tolist()
        
        if original_seqs:
            all_aa_orig = ''.join(original_seqs)
            aa_counts_orig = Counter(all_aa_orig)
            total_orig = sum(aa_counts_orig.values())
            aa_freq_orig = {aa: aa_counts_orig.get(aa, 0) / total_orig * 100 for aa in AMINO_ACIDS}
            
            print(f"\nOriginal sequences:")
            print(f"  Total residues: {total_orig:,}")
            sorted_aa_orig = sorted(aa_freq_orig.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 5 amino acids: {', '.join([f'{aa}={freq:.2f}%' for aa, freq in sorted_aa_orig[:5]])}")
    
    # ========================================================================
    # SAVE CSV RESULTS
    # ========================================================================
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    csv_file = output_dir / f"{output_prefix}_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"  ✓ CSV saved: {csv_file}")
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    # 1. Standard analysis plots (from utils.py)
    # These will be saved in output_dir/analysis_plots/
    try:
        import os
        import sys
        
        # Import generate_analysis_plots (handle both relative and absolute imports)
        try:
            from .utils import generate_analysis_plots
        except ImportError:
            # Try absolute import if relative fails
            module_dir = Path(__file__).parent
            if str(module_dir) not in sys.path:
                sys.path.insert(0, str(module_dir))
            from utils import generate_analysis_plots
        
        # Save absolute paths before changing directory
        output_dir_abs = output_dir.absolute()
        
        # Change to output_dir to ensure plots folder is created inside it
        original_dir = os.getcwd()
        os.chdir(output_dir_abs)
        generate_analysis_plots(results_df, output_prefix="analysis")
        os.chdir(original_dir)
        print(f"  ✓ Standard analysis plots saved in {output_dir_abs}/analysis_analysis_plots/")
    except Exception as e:
        print(f"  ✗ Could not generate standard plots: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Amino acid distribution plot
    try:
        # Use absolute path to avoid issues with os.chdir
        output_dir_abs = output_dir.absolute()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        aa_list = list(AMINO_ACIDS)
        recon_freq_list = [aa_freq_recon[aa] for aa in aa_list]
        
        # Plot 1: Reconstructed
        ax1 = axes[0]
        bars1 = ax1.bar(aa_list, recon_freq_list, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Frequency (%)', fontsize=12)
        ax1.set_title(f'Amino Acid Distribution - {dataset_name} (Reconstructed)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0.5:  # Only show labels for visible bars
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Comparison or just reconstructed
        ax2 = axes[1]
        if aa_freq_orig:
            orig_freq_list = [aa_freq_orig[aa] for aa in aa_list]
            x = np.arange(len(aa_list))
            width = 0.35
            
            ax2.bar(x - width/2, recon_freq_list, width, 
                   color='steelblue', alpha=0.8, label='Reconstructed')
            ax2.bar(x + width/2, orig_freq_list, width,
                   color='coral', alpha=0.8, label='Original')
            
            ax2.set_ylabel('Frequency (%)', fontsize=12)
            ax2.set_title(f'Amino Acid Distribution - Comparison', 
                         fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(aa_list)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        else:
            bars2 = ax2.bar(aa_list, recon_freq_list, color='steelblue', alpha=0.6)
            ax2.set_ylabel('Frequency (%)', fontsize=12)
            ax2.set_title(f'Amino Acid Distribution - {dataset_name}', 
                         fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        
        ax2.set_xlabel('Amino Acid', fontsize=12)
        
        plt.tight_layout()
        aa_plot_file = output_dir_abs / 'amino_acid_distribution.png'
        plt.savefig(aa_plot_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir_abs / 'amino_acid_distribution.pdf', bbox_inches='tight')
        print(f"  ✓ Amino acid distribution plot saved: {aa_plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"  ✗ Could not generate AA distribution plot: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Heatmap
    try:
        if aa_freq_orig:
            data = np.array([
                [aa_freq_recon[aa] for aa in aa_list],
                [aa_freq_orig[aa] for aa in aa_list]
            ])
            fig, ax = plt.subplots(figsize=(16, 4))
            sns.heatmap(data, annot=True, fmt='.1f', cmap='YlOrRd',
                       xticklabels=aa_list, yticklabels=['Reconstructed', 'Original'],
                       cbar_kws={'label': 'Frequency (%)'}, ax=ax)
            title = f'Amino Acid Frequency Heatmap - {dataset_name}'
        else:
            data = np.array([[aa_freq_recon[aa] for aa in aa_list]])
            fig, ax = plt.subplots(figsize=(16, 2))
            sns.heatmap(data, annot=True, fmt='.1f', cmap='YlOrRd',
                       xticklabels=aa_list, yticklabels=['Reconstructed'],
                       cbar_kws={'label': 'Frequency (%)'}, ax=ax)
            title = f'Amino Acid Frequency Heatmap - {dataset_name}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        heatmap_file = output_dir_abs / 'amino_acid_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir_abs / 'amino_acid_heatmap.pdf', bbox_inches='tight')
        print(f"  ✓ Amino acid heatmap saved: {heatmap_file}")
        plt.close()
        
    except Exception as e:
        print(f"  ✗ Could not generate AA heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # SAVE SUMMARY TEXT FILE
    # ========================================================================
    print(f"\n{'='*80}")
    print("SAVING SUMMARY")
    print(f"{'='*80}")
    
    output_dir_abs = output_dir.absolute()
    summary_file = output_dir_abs / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"FUSION RECONSTRUCTION ANALYSIS - {dataset_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total fusions: {len(results_df)}\n\n")
        
        # Length statistics
        f.write("SEQUENCE LENGTHS:\n")
        f.write(f"  Reconstructed - Mean:   {results_df['reconstructed_length'].mean():.1f} aa\n")
        f.write(f"  Reconstructed - Median: {results_df['reconstructed_length'].median():.1f} aa\n")
        f.write(f"  Reconstructed - Min:    {results_df['reconstructed_length'].min()} aa\n")
        f.write(f"  Reconstructed - Max:    {results_df['reconstructed_length'].max()} aa\n")
        
        if 'original_length' in results_df.columns:
            has_original = (results_df['original_length'] > 0).sum()
            if has_original > 0:
                f.write(f"\n  Original - Mean:   {results_df[results_df['original_length']>0]['original_length'].mean():.1f} aa\n")
                f.write(f"  Original - Median: {results_df[results_df['original_length']>0]['original_length'].median():.1f} aa\n")
                
                if 'length_diff' in results_df.columns:
                    f.write(f"\n  Length difference (reconstructed - original):\n")
                    f.write(f"    Mean:   {results_df['length_diff'].mean():.1f} aa\n")
                    f.write(f"    Median: {results_df['length_diff'].median():.1f} aa\n")
        
        # Identity statistics
        if 'identity' in results_df.columns:
            valid_identity = results_df[results_df['identity'] > 0]
            if len(valid_identity) > 0:
                f.write(f"\nSEQUENCE IDENTITY:\n")
                f.write(f"  Sequences with identity data: {len(valid_identity)}\n")
                f.write(f"  Mean:   {valid_identity['identity'].mean():.2f}%\n")
                f.write(f"  Median: {valid_identity['identity'].median():.2f}%\n")
                f.write(f"  Min:    {valid_identity['identity'].min():.2f}%\n")
                f.write(f"  Max:    {valid_identity['identity'].max():.2f}%\n")
        
        # Quality distribution
        if 'quality' in results_df.columns:
            f.write(f"\nQUALITY DISTRIBUTION:\n")
            for quality, count in results_df['quality'].value_counts().items():
                f.write(f"  {quality:<30}: {count:4d} ({100*count/len(results_df):5.1f}%)\n")
        
        # Frame analysis
        if 'in_frame' in results_df.columns:
            in_frame_count = results_df['in_frame'].sum()
            f.write(f"\nFRAME STATUS:\n")
            f.write(f"  In-frame:     {in_frame_count} ({100*in_frame_count/len(results_df):.1f}%)\n")
            f.write(f"  Out-of-frame: {len(results_df)-in_frame_count} ({100*(len(results_df)-in_frame_count)/len(results_df):.1f}%)\n")
        
        # Isoforms statistics
        if 'n_isoforms' in results_df.columns:
            f.write(f"\nISOFORM COMBINATIONS:\n")
            f.write(f"  Mean:   {results_df['n_isoforms'].mean():.1f}\n")
            f.write(f"  Median: {results_df['n_isoforms'].median():.0f}\n")
            f.write(f"  Max:    {results_df['n_isoforms'].max()}\n")
        
        # ORFfinder usage
        if 'orf_used' in results_df.columns:
            orf_count = results_df['orf_used'].sum()
            f.write(f"\nORFFINDER USAGE:\n")
            f.write(f"  Used: {orf_count} ({100*orf_count/len(results_df):.1f}%)\n")
        
        # Amino acid distribution
        f.write(f"\nAMINO ACID DISTRIBUTION (Reconstructed):\n")
        for aa in AMINO_ACIDS:
            f.write(f"  {aa}: {aa_freq_recon[aa]:.2f}%\n")
        
        if aa_freq_orig:
            f.write(f"\nAMINO ACID DISTRIBUTION (Original):\n")
            for aa in AMINO_ACIDS:
                f.write(f"  {aa}: {aa_freq_orig[aa]:.2f}%\n")
    
    print(f"  ✓ Summary saved: {summary_file}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved in: {output_dir}/")
    print(f"  - {output_prefix}_results.csv")
    print(f"  - summary.txt")
    print(f"  - amino_acid_distribution.png/pdf")
    print(f"  - amino_acid_heatmap.png/pdf")
    print(f"  - analysis_analysis_plots/ (standard plots)")
    print(f"{'='*80}")
