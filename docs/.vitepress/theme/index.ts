import DefaultTheme from "vitepress/theme";
import ImageCarousel from "../components/ImageCarousel.vue";
import PhoneMockup from "../components/PhoneMockup.vue";
import PhoneGallery from "../components/PhoneGallery.vue";
import MermaidChart from "../components/MermaidChart.vue";
import HomeLayout from "./HomeLayout.vue";
import "./custom.css";

export default {
  ...DefaultTheme,
  Layout: HomeLayout,
  enhanceApp({ app }: { app: import("vue").App }) {
    app.component("ImageCarousel", ImageCarousel);
    app.component("PhoneMockup", PhoneMockup);
    app.component("PhoneGallery", PhoneGallery);
    app.component("MermaidChart", MermaidChart);
  },
};
