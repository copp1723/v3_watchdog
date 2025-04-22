from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime


class SeverityLevel(str, Enum):
    """Severity levels for insights."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class PriorityLevel(str, Enum):
    """Priority levels for recommendations."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


SEVERITY_DEFAULT = SeverityLevel.MEDIUM
PRIORITY_DEFAULT = PriorityLevel.MEDIUM


class Insight(BaseModel):
    """Defines a concise business insight extracted from data analysis."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the insight.")
    description: str = Field(..., description="A concise description of the insight.")
    impact_area: Optional[str] = Field(None, description="The area of the dealership affected (e.g., Sales, Marketing, Inventory).")
    severity: SeverityLevel = Field(SEVERITY_DEFAULT, description="Severity level of the insight.")
    tags: List[str] = Field(default_factory=list, description="Tags to categorize or filter insights.")
    data_source: Optional[str] = Field(None, description="Source of the insight (e.g., 'CRM Analysis', 'Inventory Audit')")
    detection_method: Optional[str] = Field(None, description="Method used to detect this insight (e.g., 'Anomaly Detection', 'Statistical Comparison')")
    metric_value: Optional[float] = Field(None, description="Quantitative measure of the insight")
    metric_unit: Optional[str] = Field(None, description="Unit of the metric (e.g., '%', 'days', '$')")
    benchmark: Optional[float] = Field(None, description="Industry or historical benchmark for comparison")
    score: Optional[float] = Field(None, description="Numerical score representing urgency or value (0–1 scale).")


class Recommendation(BaseModel):
    """Defines a specific action recommended to address an insight."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the recommendation.")
    action: str = Field(..., description="A specific, actionable step to take.")
    priority: PriorityLevel = Field(PRIORITY_DEFAULT, description="Priority level for the recommendation.")
    estimated_impact: Optional[str] = Field(None, description="Potential impact of the action (e.g., Increase sales by 5%).")
    complexity: Optional[str] = Field(None, description="Implementation complexity (Easy, Medium, Difficult).")
    timeframe: Optional[str] = Field(None, description="Estimated timeframe for implementation (e.g., '1-2 weeks').")
    tags: List[str] = Field(default_factory=list, description="Tags to categorize or filter recommendations.")


class DigestEntry(BaseModel):
    """Defines a single entry in the insights digest (Problem or Opportunity)."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the digest entry.")
    title: str = Field(..., description="A concise title for the problem or opportunity.")
    detail: Insight = Field(..., description="Detailed insight describing the problem or opportunity.")
    recommendations: List[Recommendation] = Field(default_factory=list, description="A list of recommended actions to address the problem or capitalize on the opportunity.")
    tags: List[str] = Field(default_factory=list, description="Tags to categorize or filter this entry.")

    def get_priority_score(self) -> float:
        """Calculate a numerical priority score based on severity and estimated impact."""
        severity_scores = {
            SeverityLevel.HIGH: 3.0, 
            SeverityLevel.MEDIUM: 2.0, 
            SeverityLevel.LOW: 1.0
        }
        
        base_score = severity_scores.get(self.detail.severity, 1.0)
        
        # Add recommendation priority boost
        priority_scores = {
            PriorityLevel.HIGH: 3.0,
            PriorityLevel.MEDIUM: 2.0,
            PriorityLevel.LOW: 1.0
        }
        
        rec_priority = max([priority_scores.get(r.priority, 0) for r in self.recommendations]) if self.recommendations else 0
        
        return base_score + (rec_priority * 0.5)
        
    def has_tag(self, tag: str) -> bool:
        """Check if this entry has a specific tag."""
        return tag in self.tags or tag in self.detail.tags


class InsightsDigest(BaseModel):
    """Represents the complete insights digest with problems and opportunities."""
    top_problems: List[DigestEntry] = Field(default_factory=list, description="The top problems identified.")
    top_opportunities: List[DigestEntry] = Field(default_factory=list, description="The top opportunities identified.")
    generated_at: datetime = Field(default_factory=datetime.now, description="When this digest was generated")
    data_time_range: Optional[str] = Field(None, description="Time period the data covers (e.g., 'Jan-Mar 2025')")
    dealer_name: Optional[str] = Field(None, description="Name of the dealership this digest applies to")
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the digest to a JSON string."""
        return self.json(indent=indent, exclude_none=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the digest to a dictionary."""
        return self.dict(exclude_none=True)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InsightsDigest":
        """Create an InsightsDigest from a dictionary."""
        return InsightsDigest(**data)
    
    def get_all_tags(self) -> Set[str]:
        """Get all unique tags from all entries."""
        tags = set()
        
        for entry in self.top_problems + self.top_opportunities:
            tags.update(entry.tags)
            tags.update(entry.detail.tags)
            
            for rec in entry.recommendations:
                tags.update(rec.tags)
                
        return tags
    
    def filter_by_tag(self, tag: str) -> 'InsightsDigest':
        """Filter insights by tag."""
        filtered_problems = [p for p in self.top_problems if p.has_tag(tag)]
        filtered_opportunities = [o for o in self.top_opportunities if o.has_tag(tag)]
        
        return InsightsDigest(
            top_problems=filtered_problems,
            top_opportunities=filtered_opportunities,
            generated_at=self.generated_at,
            data_time_range=self.data_time_range,
            dealer_name=self.dealer_name
        )
    
    def filter_by_impact_area(self, area: str) -> 'InsightsDigest':
        """Filter insights by impact area."""
        filtered_problems = [p for p in self.top_problems if p.detail.impact_area == area]
        filtered_opportunities = [o for o in self.top_opportunities if o.detail.impact_area == area]
        
        return InsightsDigest(
            top_problems=filtered_problems,
            top_opportunities=filtered_opportunities,
            generated_at=self.generated_at,
            data_time_range=self.data_time_range,
            dealer_name=self.dealer_name
        )
    
    def to_markdown(self) -> str:
        """Convert the digest to a Markdown report."""
        md = "# Automotive Retail Insights Digest\n\n"
        
        if self.dealer_name:
            md += f"**Dealership:** {self.dealer_name}\n\n"
            
        if self.data_time_range:
            md += f"**Time Period:** {self.data_time_range}\n\n"
            
        md += f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        md += "## Top Problems\n\n"
        for problem in self.top_problems:
            md += f"### {problem.title}\n"
            md += f"**Severity**: {problem.detail.severity.value}  \n"
            md += f"**Impact Area**: {problem.detail.impact_area or 'General'}  \n\n"
            md += f"{problem.detail.description}\n\n"
            
            if problem.detail.metric_value is not None:
                md += f"**Metric**: {problem.detail.metric_value}"
                if problem.detail.metric_unit:
                    md += f"{problem.detail.metric_unit}"
                if problem.detail.benchmark is not None:
                    md += f" (Benchmark: {problem.detail.benchmark}"
                    if problem.detail.metric_unit:
                        md += f"{problem.detail.metric_unit}"
                    md += ")"
                md += "\n\n"
            
            if problem.recommendations:
                md += "**Recommendations:**\n\n"
                for rec in problem.recommendations:
                    md += f"- {rec.action} (Priority: {rec.priority.value})"
                    if rec.estimated_impact:
                        md += f" - *{rec.estimated_impact}*"
                    md += "\n"
            md += "\n"
        
        md += "## Top Opportunities\n\n"
        for opportunity in self.top_opportunities:
            md += f"### {opportunity.title}\n"
            md += f"**Severity**: {opportunity.detail.severity.value}  \n"
            md += f"**Impact Area**: {opportunity.detail.impact_area or 'General'}  \n\n"
            md += f"{opportunity.detail.description}\n\n"
            
            if opportunity.detail.metric_value is not None:
                md += f"**Metric**: {opportunity.detail.metric_value}"
                if opportunity.detail.metric_unit:
                    md += f"{opportunity.detail.metric_unit}"
                if opportunity.detail.benchmark is not None:
                    md += f" (Benchmark: {opportunity.detail.benchmark}"
                    if opportunity.detail.metric_unit:
                        md += f"{opportunity.detail.metric_unit}"
                    md += ")"
                md += "\n\n"
            
            if opportunity.recommendations:
                md += "**Recommendations:**\n\n"
                for rec in opportunity.recommendations:
                    md += f"- {rec.action} (Priority: {rec.priority.value})"
                    if rec.estimated_impact:
                        md += f" - *{rec.estimated_impact}*"
                    md += "\n"
            md += "\n"
        
        return md
    
    def get_overall_health_score(self) -> float:
        """Calculate an overall dealership health score based on problems and opportunities."""
        problem_severity = sum(3 if p.detail.severity == SeverityLevel.HIGH else 
                               2 if p.detail.severity == SeverityLevel.MEDIUM else 
                               1 for p in self.top_problems)
        
        opportunity_value = sum(3 if o.detail.severity == SeverityLevel.HIGH else 
                                2 if o.detail.severity == SeverityLevel.MEDIUM else 
                                1 for o in self.top_opportunities)
        
        # Higher is better, normalized to 0-100
        max_possible = 3 * max(10, len(self.top_problems) + len(self.top_opportunities))
        return min(100, max(0, 100 - (problem_severity * 100 / max_possible) + (opportunity_value * 50 / max_possible)))


def create_insights_digest(problems: List[Dict[str, Any]], opportunities: List[Dict[str, Any]]) -> InsightsDigest:
    """
    Creates an InsightsDigest from lists of problems and opportunities (represented as dictionaries).
    This function handles the conversion from dictionaries to Pydantic models, adapting to both 'insight' and 'detail' field names.

    Args:
        problems: A list of dictionaries, where each dictionary represents a problem.
        opportunities: A list of dictionaries, where each dictionary represents an opportunity.

    Returns:
        An InsightsDigest object containing the top problems and opportunities.
    """

    def create_digest_entry(data: Dict[str, Any]) -> DigestEntry:
        """Helper function to create a DigestEntry from a dictionary."""
        # Handle both 'insight' and 'detail' field names for backward compatibility
        insight_data = data.get('detail', data.get('insight', {}))
        insight = Insight(**insight_data)

        # Create Recommendation objects
        recommendation_data = data.get('recommendations', [])
        recommendations = [Recommendation(**rec) for rec in recommendation_data]

        # Extract tags
        tags = data.get('tags', [])

        return DigestEntry(
            title=data.get('title', 'Untitled'),
            detail=insight,
            recommendations=recommendations,
            tags=tags
        )

    top_problems = [create_digest_entry(problem) for problem in problems]
    top_opportunities = [create_digest_entry(opportunity) for opportunity in opportunities]

    return InsightsDigest(top_problems=top_problems, top_opportunities=top_opportunities)


# Example Usage (Outside the Class - For Testing)
if __name__ == '__main__':
    # Sample data (using dictionaries for flexibility)
    sample_problems = [
        {
            "title": "High Inventory Aging",
            "detail": {
                "description": "25% of inventory is over 120 days old, leading to potential losses.",
                "impact_area": "Inventory",
                "severity": "High",
                "data_source": "Inventory System",
                "detection_method": "Aging Analysis",
                "metric_value": 25.0,
                "metric_unit": "%",
                "benchmark": 10.0,
                "tags": ["inventory", "aging"]
            },
            "recommendations": [
                {
                    "action": "Implement aggressive markdown pricing on aged inventory.",
                    "priority": "High",
                    "estimated_impact": "Reduce aged inventory by 15% within 30 days.",
                    "tags": ["pricing", "markdown"]
                },
                {
                    "action": "Offer sales incentives for moving aged inventory.",
                    "priority": "Medium",
                    "tags": ["sales", "incentives"]
                }
            ],
            "tags": ["inventory", "aging"]
        },
        {
            "title": "Low Lead Conversion Rate",
            "detail": {
                "description": "CRM data shows a lead conversion rate of only 2%, significantly below the industry average.",
                "impact_area": "Sales",
                "severity": "High",
                "data_source": "CRM",
                "detection_method": "Conversion Rate Analysis",
                "metric_value": 2.0,
                "metric_unit": "%",
                "benchmark": 5.0,
                "tags": ["sales", "crm"]
            },
            "recommendations": [
                {
                    "action": "Review and optimize lead nurturing processes.",
                    "priority": "High",
                    "tags": ["sales", "crm", "nurturing"]
                },
                {
                    "action": "Provide additional sales training on lead qualification.",
                    "priority": "Medium",
                    "tags": ["sales", "training"]
                }
            ],
            "tags": ["lead", "crm"]
        }
    ]

    sample_opportunities = [
        {
            "title": "Untapped Service Upselling Potential",
            "detail": {
                "description": "Analysis of service records reveals a low upselling rate for maintenance packages.",
                "impact_area": "Service",
                "severity": "Medium",
                "data_source": "Service Records",
                "detection_method": "Upselling Analysis",
                "metric_value": 5.0,
                "metric_unit": "%",
                "benchmark": 10.0,
                "tags": ["service", "upselling"]
            },
            "recommendations": [
                {
                    "action": "Implement a service advisor training program focused on upselling techniques.",
                    "priority": "Medium",
                    "tags": ["service", "training"]
                },
                {
                    "action": "Offer bundled service packages with attractive pricing.",
                    "priority": "Medium",
                    "tags": ["service", "pricing", "bundles"]
                }
            ],
            "tags": ["service"]
        }
    ]

    # Create the InsightsDigest
    digest = create_insights_digest(sample_problems, sample_opportunities)

    # Print the JSON output
    print("JSON Output:")
    print(digest.to_json())

    # Print the Markdown output
    print("\nMarkdown Output:")
    print(digest.to_markdown())

    # Filter by impact area
    filtered_digest = digest.filter_by_impact_area("Sales")
    print("\nFiltered by Sales:")
    print(filtered_digest.to_json())

    # Get the overall health score
    health_score = digest.get_overall_health_score()
    print(f"\nOverall Health Score: {health_score:.2f}")
