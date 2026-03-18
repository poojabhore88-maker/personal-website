// Mobile menu toggle
const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const mobileMenu = document.getElementById('mobile-menu');

mobileMenuBtn.addEventListener('click', () => {
    mobileMenu.classList.toggle('hidden');
});

// Close mobile menu when clicking on a link
const mobileLinks = mobileMenu.querySelectorAll('a');
mobileLinks.forEach(link => {
    link.addEventListener('click', () => {
        mobileMenu.classList.add('hidden');
    });
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add scroll effect to navigation
window.addEventListener('scroll', () => {
    const nav = document.querySelector('nav');
    if (window.scrollY > 50) {
        nav.classList.add('shadow-md');
    } else {
        nav.classList.remove('shadow-md');
    }
});

// Form submission handler
document.getElementById('contact-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    
    // Create SMS message
    const message = `New Contact Form Submission:\nName: ${data.name}\nEmail: ${data.email}\nSubject: ${data.subject}\nMessage: ${data.message}`;
    
    // Create SMS link
    const smsUrl = `sms:+919028458981?body=${encodeURIComponent(message)}`;
    
    // Open SMS app
    window.location.href = smsUrl;
    
    // Show success message
    showNotification('Opening messaging app to send your message...', 'success');
    
    // Reset form
    this.reset();
});

// Notification function
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-20 right-6 px-6 py-4 rounded-lg shadow-lg z-50 transform translate-x-full transition-transform duration-300`;
    
    // Set background color based on type
    if (type === 'success') {
        notification.classList.add('bg-green-500', 'text-white');
    } else if (type === 'error') {
        notification.classList.add('bg-red-500', 'text-white');
    } else {
        notification.classList.add('bg-blue-500', 'text-white');
    }
    
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'} mr-3"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full');
        notification.classList.add('translate-x-0');
    }, 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.classList.remove('translate-x-0');
        notification.classList.add('translate-x-full');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-fadeInUp');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('section').forEach(section => {
    observer.observe(section);
});

// Add typing effect to hero section
function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.innerHTML = '';
    
    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Initialize typing effect when page loads
window.addEventListener('load', () => {
    const heroTitle = document.querySelector('#home h1');
    if (heroTitle) {
        const originalText = heroTitle.textContent;
        typeWriter(heroTitle, originalText, 150);
    }
});

// Add parallax effect to hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('#home');
    if (hero) {
        hero.style.transform = `translateY(${scrolled * 0.5}px)`;
    }
});

// Add hover effect to skill cards
document.querySelectorAll('.card-hover').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Add active state to navigation based on scroll position
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (scrollY >= (sectionTop - 200)) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('text-purple-600');
        if (link.getAttribute('href').slice(1) === current) {
            link.classList.add('text-purple-600');
        }
    });
});

// GitHub Repositories functionality
let repositoriesData = [];

async function loadRepositories() {
    const loadingEl = document.getElementById('loading-repos');
    const errorEl = document.getElementById('error-repos');
    const gridEl = document.getElementById('repositories-grid');
    
    // Show loading state
    loadingEl.classList.remove('hidden');
    errorEl.classList.add('hidden');
    gridEl.classList.add('hidden');
    
    try {
        // Replace 'poojakokate' with your actual GitHub username
        const username = 'poojabhore88-maker';
        const response = await fetch(`https://api.github.com/users/${username}/repos?sort=updated&per_page=9`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch repositories');
        }
        
        const repos = await response.json();
        repositoriesData = repos;
        displayRepositories(repos);
        
        // Hide loading, show grid
        loadingEl.classList.add('hidden');
        gridEl.classList.remove('hidden');
        
    } catch (error) {
        console.error('Error loading repositories:', error);
        loadingEl.classList.add('hidden');
        errorEl.classList.remove('hidden');
    }
}

function displayRepositories(repos) {
    const gridEl = document.getElementById('repositories-grid');
    
    if (repos.length === 0) {
        gridEl.innerHTML = `
            <div class="col-span-full text-center py-12">
                <i class="fas fa-folder-open text-4xl text-gray-400 mb-4"></i>
                <p class="text-gray-600">No repositories found.</p>
            </div>
        `;
        return;
    }
    
    gridEl.innerHTML = repos.map(repo => createRepositoryCard(repo)).join('');
}

function createRepositoryCard(repo) {
    const description = repo.description || 'No description available';
    const language = repo.language || 'Unknown';
    const stars = repo.stargazers_count;
    const forks = repo.forks_count;
    const updatedAt = new Date(repo.updated_at).toLocaleDateString();
    
    return `
        <div class="card-hover bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden">
            <div class="p-6">
                <div class="flex items-start justify-between mb-4">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                            <i class="fab fa-github text-purple-600"></i>
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800 text-lg">${repo.name}</h3>
                            <span class="text-xs text-gray-500">Updated ${updatedAt}</span>
                        </div>
                    </div>
                </div>
                
                <p class="text-gray-600 text-sm mb-4 line-clamp-3">${description}</p>
                
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center space-x-2">
                        ${language !== 'Unknown' ? `
                            <span class="inline-block w-3 h-3 rounded-full" style="background-color: ${getLanguageColor(language)}"></span>
                            <span class="text-xs text-gray-600">${language}</span>
                        ` : ''}
                    </div>
                    <div class="flex items-center space-x-4 text-xs text-gray-500">
                        <span class="flex items-center">
                            <i class="fas fa-star mr-1"></i>
                            ${stars}
                        </span>
                        <span class="flex items-center">
                            <i class="fas fa-code-branch mr-1"></i>
                            ${forks}
                        </span>
                    </div>
                </div>
                
                <div class="flex space-x-2">
                    <a href="${repo.html_url}" target="_blank" rel="noopener noreferrer"
                       class="flex-1 px-3 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 transition-colors text-center">
                        <i class="fas fa-external-link-alt mr-1"></i>
                        View
                    </a>
                    ${repo.homepage ? `
                        <a href="${repo.homepage}" target="_blank" rel="noopener noreferrer"
                           class="px-3 py-2 border border-gray-300 text-gray-700 text-sm rounded-lg hover:bg-gray-50 transition-colors">
                            <i class="fas fa-globe mr-1"></i>
                            Demo
                        </a>
                    ` : ''}
                </div>
            </div>
        </div>
    `;
}

function getLanguageColor(language) {
    const colors = {
        'JavaScript': '#f1e05a',
        'TypeScript': '#2b7489',
        'Python': '#3572A5',
        'Java': '#b07219',
        'C++': '#f34b7d',
        'C#': '#239120',
        'PHP': '#4F5D95',
        'Ruby': '#701516',
        'Go': '#00ADD8',
        'Rust': '#dea584',
        'Swift': '#ffac45',
        'Kotlin': '#F18E33',
        'HTML': '#e34c26',
        'CSS': '#563d7c',
        'Vue': '#41b883',
        'React': '#61dafb',
        'Angular': '#dd0031'
    };
    return colors[language] || '#858585';
}

// Load repositories when the repositories section comes into view
const reposObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && repositoriesData.length === 0) {
            loadRepositories();
            reposObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

// Start observing the repositories section when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const reposSection = document.getElementById('repositories');
    if (reposSection) {
        reposObserver.observe(reposSection);
    }
});
